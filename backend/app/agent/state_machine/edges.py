"""状态机公共逻辑：确认/取消语义判断 + state 订单还原。"""
from datetime import datetime
from typing import Any, Dict

from app.services.models import OrderInfo, OrderItem

CONFIRM_WORDS = {"确认", "确定", "好的", "好", "行", "可以", "是的", "对"}
DENY_WORDS = {"取消", "不要了", "算了", "不退了", "不用了", "撤销", "停下", "退出"}


def is_confirm(text: str) -> bool:
    t = text.strip().lower()
    return t in CONFIRM_WORDS or t.startswith("确认")


def is_deny(text: str) -> bool:
    t = text.strip().lower()
    return any(t == w or t.startswith(w) for w in DENY_WORDS)


def order_from_state(state: Dict[str, Any]) -> OrderInfo:
    """把 state 里缓存的订单 dict 还原为 OrderInfo。"""
    d = state["order"]
    return OrderInfo(
        order_id=d["order_id"],
        user_id=state["user_id"],
        status=d["status"],
        total_amount=d["total_amount"],
        shipping_address=d.get("shipping_address", ""),
        delivered_at=datetime.fromisoformat(d["delivered_at"]) if d.get("delivered_at") else None,
        db_id=d.get("db_id", 0),
        items=[
            OrderItem(
                id=i.get("id", 0),
                item_id=i["item_id"],
                name=i["name"],
                price=i["price"],
                quantity=i["quantity"],
                returnable=i.get("returnable", True),
                status=i.get("status", "NORMAL"),
            )
            for i in d.get("items", [])
        ],
    )
