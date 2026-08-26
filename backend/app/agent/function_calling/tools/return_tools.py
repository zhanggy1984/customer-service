"""退货工具。内部调用对接层 IReturnService。"""
from app.services import order_service, return_service


async def check_return_eligibility(params: dict, user_id: int, session_id: str) -> dict:
    order_id = params.get("order_id")
    if not order_id:
        return {"ok": False, "data": None, "error": {"code": "missing_order_id", "message": "缺少 order_id 参数"}}
    order = await order_service.query_order(order_id, user_id)
    if not order:
        return {"ok": False, "data": None, "error": {"code": "order_not_found", "message": "订单不存在或不属于当前用户"}}
    result = await return_service.check_eligibility(order, user_id)
    return {
        "ok": True,
        "data": {
            "eligible": result["eligible"],
            "reason": result.get("reason", ""),
            "refund_amount": result.get("refund_amount", 0),
            "items": [
                {"item_id": i.item_id, "name": i.name, "price": i.price, "quantity": i.quantity}
                for i in result.get("items", [])
            ],
        },
        "error": None,
    }


async def create_return_order(params: dict, user_id: int, session_id: str) -> dict:
    order_id = params.get("order_id")
    items = params.get("items", [])
    reason = params.get("reason", "")
    if not order_id or not items:
        return {"ok": False, "data": None, "error": {"code": "missing_order_items", "message": "缺少 order_id 或 items 参数"}}
    order = await order_service.query_order(order_id, user_id)
    if not order:
        return {"ok": False, "data": None, "error": {"code": "order_not_found", "message": "订单不存在或不属于当前用户"}}
    result = await return_service.create_return(order, user_id, items, reason, session_id)
    return {
        "ok": True,
        "data": {
            "success": result.success,
            "status": result.status,
            "return_id": result.return_id,
            "refund_amount": result.refund_amount,
            "message": result.message,
        },
        "error": None,
    }
