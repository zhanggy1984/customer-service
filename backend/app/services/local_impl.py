"""对接层 Local 实现：直接操作 MySQL 模拟微服务。

升级路径: Agent → IOrderService → LocalOrderService → MySQL
未来:     Agent → IOrderService → RemoteOrderService → HTTP/gRPC → 订单微服务
"""
import hashlib
import json
import time
from datetime import datetime

from asyncmy.errors import IntegrityError

from app.infrastructure.mysql import mysql_pool
from app.services.interfaces import (
    IComplaintService,
    IOrderService,
    IRefundService,
    IReturnService,
    ITicketService,
)
from app.services.exceptions import ServiceUnavailableException
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


def _ticket_idempotency_key(user_id: int, order_id: str | None, complaint_type: str, description: str) -> str:
    """投诉幂等键：用户+订单+类型+描述整体哈希派生的定长键（CT:+32 位 hex=36 ≤ VARCHAR(64)）。

    order_id/complaint_type 最长 32 字符，明文拼接可超 64 触发截断（严格模式报错/非严格截断
    导致幂等语义错乱），故对拼接串整体 sha256 定长化；确定性保留（同参数同键）。
    重试（DB 已提交但响应前断开）与重复提交同一内容时键相同，uk_ticket_idempotency 冲突 →
    不重复建单。同一内容重复投诉被幂等合并（防误操作）；新内容摘要不同正常新建。
    """
    material = f"{user_id}:{order_id or '-'}:{complaint_type or '-'}:{description or ''}"
    return f"CT:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _manual_idempotency_key(user_id: int, message: str) -> str:
    """转人工幂等键：用户+消息整体哈希定长（复用 complaint_tickets 的 uk_ticket_idempotency）。"""
    material = f"{user_id}:{message or ''}"
    return f"MT:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


class LocalOrderService(IOrderService):
    @_retry_on_db_error
    async def query_order(self, order_id: str, user_id: int) -> OrderInfo | None:
        row = await mysql_pool.fetchone("SELECT * FROM orders WHERE order_id=%s", (order_id,))
        # 订单不存在或不属于当前用户 → 返回 None（用户隔离由这里保证）
        if not row or row["user_id"] != user_id:
            return None
        return _row_to_order(row, await _load_items(row["id"]))

    @_retry_on_db_error
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
        try:
            async with mysql_pool.transaction() as run:
                await run(
                    "INSERT INTO return_orders "
                    "(return_id, order_id, user_id, items, reason, refund_amount, status, session_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,'APPROVED',%s)",
                    (return_id, order.db_id, user_id, json.dumps(items_payload, ensure_ascii=False),
                     reason, refund_amount, session_id),
                )
                for it in eligible:
                    # 只更新仍为 NORMAL 的商品，防重复退
                    await run(
                        "UPDATE order_items SET status='RETURNED' WHERE order_id=%s AND item_id=%s AND status='NORMAL'",
                        (order.db_id, it.item_id),
                    )
        except IntegrityError:
            # transaction 已 ROLLBACK；唯一键冲突（uk_return_order_user）：重试/重复退货 → 幂等返回已有单号。
            # 不用 INSERT IGNORE：它把外键/类型错误也吞成 rowcount=0，误判为幂等冲突造成假拒绝。
            row = await mysql_pool.fetchone(
                "SELECT return_id FROM return_orders WHERE order_id=%s AND user_id=%s",
                (order.db_id, user_id),
            )
            if not row:
                # 非唯一键冲突（return_id 时间戳碰撞/外键归入 IntegrityError）→ 不回落假成功
                raise ServiceUnavailableException("退货提交失败，请稍后重试") from None
            existing = row["return_id"]
            logger.info("event=return_idempotent", extra={"order_id": order.order_id, "return_id": existing})
            return ReturnResult(
                success=True,
                return_id=existing,
                status="APPROVED",
                refund_amount=refund_amount,
                message="您已提交过退货申请，单号 " + existing,
            )
        logger.info(
            "event=return_created",
            extra={"order_id": order.order_id, "return_id": return_id, "amount": refund_amount},
        )
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
        try:
            await mysql_pool.execute(
                "INSERT INTO refund_orders (refund_id, order_id, user_id, reason, amount, status, session_id) "
                "VALUES (%s,%s,%s,%s,%s,'APPROVED',%s)",
                (refund_id, order.db_id, user_id, reason, amount, session_id),
            )
        except IntegrityError:
            # 唯一键冲突（uk_refund_order_user）：DB 已提交但响应前断开的重试/重复提交
            # → 返回已存在的退款单号，不重复创建。不用 INSERT IGNORE：它把外键/类型错误也
            # 吞成 rowcount=0，误判为幂等冲突造成假成功。
            row = await mysql_pool.fetchone(
                "SELECT refund_id FROM refund_orders WHERE order_id=%s AND user_id=%s",
                (order.db_id, user_id),
            )
            if not row:
                # 非唯一键冲突（refund_id 时间戳碰撞/外键归入 IntegrityError）→ 不回落假成功
                raise ServiceUnavailableException("退款提交失败，请稍后重试") from None
            existing = row["refund_id"]
            logger.info("event=refund_idempotent", extra={"order_id": order.order_id, "refund_id": existing})
            return RefundResult(
                success=True,
                refund_id=existing,
                status="APPROVED",
                amount=amount,
                message="您已提交过退款申请，单号 " + existing,
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
        key = _ticket_idempotency_key(user_id, order_id, complaint_type, description)
        try:
            await mysql_pool.execute(
                "INSERT INTO complaint_tickets "
                "(ticket_id, user_id, order_id, complaint_type, description, severity, status, session_id, idempotency_key) "
                "VALUES (%s,%s,%s,%s,%s,%s,'OPEN',%s,%s)",
                (ticket_id, user_id, order_id, complaint_type, description, severity, session_id, key),
            )
        except IntegrityError:
            # 唯一键冲突（uk_ticket_idempotency）：重试/重复提交同一投诉 → 返回已存在的工单号
            row = await mysql_pool.fetchone(
                "SELECT ticket_id FROM complaint_tickets WHERE idempotency_key=%s", (key,)
            )
            if not row:
                # 非唯一键冲突（ticket_id 时间戳碰撞）→ 不回落假成功
                raise ServiceUnavailableException("投诉提交失败，请稍后重试") from None
            existing = row["ticket_id"]
            logger.info("event=complaint_idempotent", extra={"ticket_id": existing, "severity": severity})
            return ComplaintResult(
                success=True, ticket_id=existing, severity=severity,
                message="您已提交过该投诉，工单号 " + existing,
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
        key = _manual_idempotency_key(user_id, message)
        try:
            await mysql_pool.execute(
                "INSERT INTO complaint_tickets "
                "(ticket_id, user_id, order_id, complaint_type, description, severity, status, session_id, idempotency_key) "
                "VALUES (%s,%s,%s,'MANUAL',%s,'MEDIUM','OPEN',%s,%s)",
                (ticket_id, user_id, order_id, message, session_id, key),
            )
        except IntegrityError:
            # 唯一键冲突（uk_ticket_idempotency）：同一转人工消息重复提交（重试/重复触发）→ 返回已存在工单号
            row = await mysql_pool.fetchone(
                "SELECT ticket_id FROM complaint_tickets WHERE idempotency_key=%s", (key,)
            )
            if not row:
                raise ServiceUnavailableException("转人工提交失败，请稍后重试") from None
            existing = row["ticket_id"]
            logger.info("event=manual_ticket_idempotent", extra={"ticket_id": existing})
            return ManualTicket(ticket_id=existing, message="已转人工客服，工单号 " + existing)
        logger.info("event=manual_ticket_created", extra={"ticket_id": ticket_id})
        return ManualTicket(ticket_id=ticket_id, message="已转人工客服，工单号 " + ticket_id)
