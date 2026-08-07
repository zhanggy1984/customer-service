"""订单工具。内部调用对接层 IOrderService。"""
from app.services import order_service


def _order_to_dict(order) -> dict:
    return {
        "db_id": order.db_id,  # 内部主键：状态机创建退货单时用于更新 order_items
        "order_id": order.order_id,
        "status": order.status,
        "total_amount": order.total_amount,
        "shipping_address": order.shipping_address,
        "created_at": order.created_at.isoformat() if order.created_at else None,
        "delivered_at": order.delivered_at.isoformat() if order.delivered_at else None,
        "items": [
            {
                "item_id": i.item_id,
                "name": i.name,
                "price": i.price,
                "quantity": i.quantity,
                "returnable": i.returnable,
                "status": i.status,
            }
            for i in order.items
        ],
    }


async def query_order(params: dict, user_id: int, session_id: str) -> dict:
    order_id = params.get("order_id")
    if not order_id:
        return {"error": "缺少 order_id 参数"}
    order = await order_service.query_order(order_id, user_id)
    if not order:
        return {"not_found": True, "message": "订单不存在或不属于当前用户"}
    return {"order": _order_to_dict(order)}


async def list_user_orders(params: dict, user_id: int, session_id: str) -> dict:
    try:
        limit = int(params.get("limit", 5))
    except (TypeError, ValueError):
        limit = 5
    orders = await order_service.list_user_orders(user_id, limit=limit)
    if not orders:
        return {"orders": [], "message": "您最近没有订单"}
    return {"orders": [_order_to_dict(o) for o in orders]}
