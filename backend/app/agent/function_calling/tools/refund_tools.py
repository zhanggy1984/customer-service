"""仅退款工具。内部调用对接层 IRefundService。"""
from app.services import order_service, refund_service


async def create_refund_order(params: dict, user_id: int, session_id: str) -> dict:
    order_id = params.get("order_id")
    reason = params.get("reason", "")
    if not order_id:
        return {"error": "缺少 order_id 参数"}
    order = await order_service.query_order(order_id, user_id)
    if not order:
        return {"not_found": True, "message": "订单不存在或不属于当前用户"}
    # 先做三级资格判定，不通过则直接返回拒绝原因
    elig = await refund_service.check_refund_eligibility(order)
    if not elig["eligible"]:
        return {
            "success": False,
            "status": "REJECTED",
            "message": elig["reason"],
        }
    result = await refund_service.create_refund(order, user_id, reason, session_id)
    return {
        "success": result.success,
        "status": result.status,
        "refund_id": result.refund_id,
        "amount": result.amount,
        "message": result.message,
    }
