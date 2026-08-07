"""对接层 Local 实现：直接操作 MySQL 模拟微服务。

升级路径: Agent → IOrderService → LocalOrderService → MySQL
未来:     Agent → IOrderService → RemoteOrderService → HTTP/gRPC → 订单微服务
"""
import json
import time
from datetime import datetime

from app.infrastructure.mysql import mysql_pool
from app.services.interfaces import (
    IComplaintService,
    IOrderService,
    IRefundService,
    IReturnService,
    ITicketService,
)
from app.services.retry import _retry_on_db_error
from app.services.models import (
    ComplaintResult,
    ManualTicket,
    OrderInfo,
    OrderItem,
    RefundResult,
    ReturnResult,
)
from app.utils.logger import logger


def _row_to_order_item(row: dict) -> OrderItem:
    return OrderItem(
        id=row["id"],
        item_id=row["item_id"],
        name=row["name"],
        price=float(row["price"]),
        quantity=row["quantity"],
        returnable=bool(row["returnable"]),
        status=row["status"],
    )


def _row_to_order(row: dict, items: list[OrderItem]) -> OrderInfo:
    return OrderInfo(
        order_id=row["order_id"],
        user_id=row["user_id"],
        status=row["status"],
        total_amount=float(row["total_amount"]),
        shipping_address=row["shipping_address"] or "",
        created_at=row["created_at"],
        delivered_at=row["delivered_at"],
        db_id=row["id"],
        items=items,
    )


async def _load_items(order_db_id: int) -> list[OrderItem]:
    rows = await mysql_pool.fetchall(
        "SELECT * FROM order_items WHERE order_id=%s ORDER BY id", (order_db_id,)
    )
    return [_row_to_order_item(r) for r in rows]


def _refundable_amount(order: OrderInfo) -> tuple[float, list[OrderItem]]:
    """可退款商品合计：过滤定制类(returnable=false)与已退过(status!=NORMAL)的商品。

    仅退款的资格判定(check)与落库(create)共用同一口径，避免两处金额不一致。
    返回 (金额, 可退商品列表)；金额为 0 且列表为空表示无可退商品。
    """
    eligible = [it for it in order.items if it.returnable and it.status == "NORMAL"]
    return round(sum(it.price * it.quantity for it in eligible), 2), eligible


class LocalOrderService(IOrderService):
    async def query_order(self, order_id: str, user_id: int) -> OrderInfo | None:
        row = await mysql_pool.fetchone("SELECT * FROM orders WHERE order_id=%s", (order_id,))
        # 订单不存在或不属于当前用户 → 返回 None（用户隔离由这里保证）
        if not row or row["user_id"] != user_id:
            return None
        return _row_to_order(row, await _load_items(row["id"]))

    async def list_user_orders(self, user_id: int, limit: int = 5) -> list[OrderInfo]:
        rows = await mysql_pool.fetchall(
            "SELECT * FROM orders WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
        return [_row_to_order(r, await _load_items(r["id"])) for r in rows]


class LocalReturnService(IReturnService):
    async def check_eligibility(self, order: OrderInfo, user_id: int) -> dict:
        if not order.is_owner(user_id):
            return {"eligible": False, "reason": "该订单不属于您的账号", "refund_amount": 0.0, "items": []}
        if order.status == "CANCELLED":
            return {"eligible": False, "reason": "订单已取消，无法操作", "refund_amount": 0.0, "items": []}
        if order.status == "PAID":
            return {"eligible": False, "reason": "订单未发货，建议申请仅退款", "refund_amount": 0.0, "items": []}
        if order.status == "SHIPPED":
            return {"eligible": False, "reason": "订单运输中，请先拒收或等待签收后走退货", "refund_amount": 0.0, "items": []}
        if order.delivered_at:
            days = (datetime.now() - order.delivered_at).days
            if days > 7:
                return {"eligible": False, "reason": "已超过 7 天退货期，不可退", "refund_amount": 0.0, "items": []}
        eligible_items = [it for it in order.items if it.returnable and it.status == "NORMAL"]
        if not eligible_items:
            return {"eligible": False, "reason": "该商品不支持退货或已退过", "refund_amount": 0.0, "items": []}
        refund_amount = round(sum(it.price * it.quantity for it in eligible_items), 2)
        return {"eligible": True, "reason": "", "refund_amount": refund_amount, "items": eligible_items}

    @_retry_on_db_error
    async def create_return(
        self, order: OrderInfo, user_id: int, items: list[str], reason: str, session_id: str
    ) -> ReturnResult:
        eligible = [it for it in order.items if it.item_id in items]
        if not eligible:
            return ReturnResult(success=False, status="REJECTED", message="所选商品不在订单中或不可退")
        refund_amount = round(sum(it.price * it.quantity for it in eligible), 2)
        # return_id 用短格式（RC-时间戳），避免超 VARCHAR(32)；order_id 用 DB 主键（FK 整数）
        return_id = f"RC-{int(time.time() * 1000)}"
        items_payload = [
            {
                "item_id": it.item_id,
                "name": it.name,
                "quantity": it.quantity,
                "refund": round(it.price * it.quantity, 2),
            }
            for it in eligible
        ]
        async with mysql_pool.transaction() as run:
            cur = await run(
                "INSERT IGNORE INTO return_orders "
                "(return_id, order_id, user_id, items, reason, refund_amount, status, session_id) "
                "VALUES (%s,%s,%s,%s,%s,%s,'APPROVED',%s)",
                (return_id, order.db_id, user_id, json.dumps(items_payload, ensure_ascii=False),
                 reason, refund_amount, session_id),
            )
            inserted = cur.rowcount > 0
            if inserted:
                for it in eligible:
                    # 只更新仍为 NORMAL 的商品，防重复退
                    await run(
                        "UPDATE order_items SET status='RETURNED' WHERE order_id=%s AND item_id=%s AND status='NORMAL'",
                        (order.db_id, it.item_id),
                    )
        logger.info(
            "event=return_created",
            extra={"order_id": order.order_id, "return_id": return_id, "amount": refund_amount, "inserted": inserted},
        )
        if not inserted:
            return ReturnResult(success=False, status="REJECTED", message="该订单已提交过退货申请")
        return ReturnResult(
            success=True,
            return_id=return_id,
            status="APPROVED",
            refund_amount=refund_amount,
            message="退货单已创建",
        )


class LocalRefundService(IRefundService):
    async def check_refund_eligibility(self, order: OrderInfo) -> dict:
        if order.status == "PAID":
            amount, eligible_items = _refundable_amount(order)
            if not eligible_items:
                return {"eligible": False, "reason": "订单内商品均为定制类，不支持退款", "amount": 0.0}
            return {"eligible": True, "reason": "未发货，可申请仅退款", "amount": amount}
        if order.status == "SHIPPED":
            return {"eligible": False, "reason": "订单已发货，请先拒收，物流退回后自动退款", "amount": 0.0}
        if order.status == "DELIVERED":
            return {"eligible": False, "reason": "订单已签收，仅退款不适用，请走退货流程", "amount": 0.0}
        return {"eligible": False, "reason": "订单当前状态不支持仅退款", "amount": 0.0}

    @_retry_on_db_error
    async def create_refund(
        self, order: OrderInfo, user_id: int, reason: str, session_id: str
    ) -> RefundResult:
        refund_id = f"RF-{int(time.time() * 1000)}"
        # 金额与 check_refund_eligibility 同口径（过滤定制类商品），避免判定与落库不一致
        amount, _ = _refundable_amount(order)
        await mysql_pool.execute(
            "INSERT INTO refund_orders (refund_id, order_id, user_id, reason, amount, status, session_id) "
            "VALUES (%s,%s,%s,%s,%s,'APPROVED',%s)",
            (refund_id, order.db_id, user_id, reason, amount, session_id),
        )
        logger.info("event=refund_created", extra={"order_id": order.order_id, "refund_id": refund_id})
        return RefundResult(
            success=True,
            refund_id=refund_id,
            status="APPROVED",
            amount=amount,
            message="退款申请已提交",
        )


class LocalComplaintService(IComplaintService):
    @_retry_on_db_error
    async def create_complaint(
        self,
        user_id: int,
        order_id: str | None,
        complaint_type: str,
        description: str,
        severity: str,
        session_id: str,
    ) -> ComplaintResult:
        ticket_id = f"CT-{int(time.time() * 1000)}"  # 3+13=16 字符，VARCHAR(32) 内
        await mysql_pool.execute(
            "INSERT INTO complaint_tickets "
            "(ticket_id, user_id, order_id, complaint_type, description, severity, status, session_id) "
            "VALUES (%s,%s,%s,%s,%s,%s,'OPEN',%s)",
            (ticket_id, user_id, order_id, complaint_type, description, severity, session_id),
        )
        logger.info("event=complaint_created", extra={"ticket_id": ticket_id, "severity": severity})
        return ComplaintResult(success=True, ticket_id=ticket_id, severity=severity, message="投诉工单已创建")


class LocalTicketService(ITicketService):
    @_retry_on_db_error
    async def create_manual_ticket(
        self,
        user_id: int,
        order_id: str | None,
        session_id: str,
        message: str,
    ) -> ManualTicket:
        ticket_id = f"MT-{int(time.time() * 1000)}"
        await mysql_pool.execute(
            "INSERT INTO complaint_tickets "
            "(ticket_id, user_id, order_id, complaint_type, description, severity, status, session_id) "
            "VALUES (%s,%s,%s,'MANUAL',%s,'MEDIUM','OPEN',%s)",
            (ticket_id, user_id, order_id, message, session_id),
        )
        logger.info("event=manual_ticket_created", extra={"ticket_id": ticket_id})
        return ManualTicket(ticket_id=ticket_id, message="已转人工客服，工单号 " + ticket_id)
