"""退货状态机（LangGraph）。

节点链: collect_order_id → verify_order → check_eligibility → collect_reason → confirm → execute → notify
执行模型: 每轮用户输入推进一个节点（见 base.py）。
全局打断: 任意输入节点识别到"取消/不要了"等语义 → 直接 END。
"""
from datetime import datetime
from typing import TypedDict

from langgraph.graph import END

from app.agent.function_calling.tools.order_tools import _order_to_dict
from app.agent.state_machine.base import BaseStateMachine
from app.services import order_service, return_service
from app.services.models import OrderInfo, OrderItem

CONFIRM_WORDS = {"确认", "确定", "好的", "好", "行", "可以", "是的", "对"}
DENY_WORDS = {"取消", "不要了", "算了", "不退了", "不用了", "撤销", "停下", "退出"}


class ReturnState(TypedDict, total=False):
    user_id: int
    session_id: str
    order_id: str
    return_items: list  # 用户指定只退的部分商品名数组；空 = 退全部可退商品
    order: dict
    reason: str
    confirmed: bool
    eligibility: dict
    result: dict
    user_input: str
    stage: str
    message: str
    awaiting: str
    final: bool
    tool_calls: list  # 契约透出：query_order/create_return 动作（orchestrator 取出 emit tool_call）


def _is_confirm(text: str) -> bool:
    t = text.strip().lower()
    return t in CONFIRM_WORDS or t.startswith("确认")


def _is_deny(text: str) -> bool:
    t = text.strip().lower()
    return any(t == w or t.startswith(w) for w in DENY_WORDS)


def _to_order(state: dict) -> OrderInfo:
    """把 state 里的订单 dict 还原为 OrderInfo 领域对象。"""
    d = state["order"]
    return OrderInfo(
        order_id=d["order_id"],
        user_id=state["user_id"],  # user_id 是会话级字段，不在订单 dict 中
        status=d["status"],
        total_amount=d["total_amount"],
        shipping_address=d.get("shipping_address", ""),
        # delivered_at 参与超期判定，必须从 dict 还原（否则退货资格检查失效）
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


async def _collect_order_id(state):
    if state.get("order_id"):
        return {"stage": "verify_order", "awaiting": None}
    oid = (state.get("user_input") or "").strip()
    if state.get("awaiting") == "order_id" and oid:
        return {"order_id": oid, "awaiting": None, "stage": "verify_order"}
    return {"stage": "collect_order_id", "awaiting": "order_id", "message": "请提供您的订单号"}


async def _verify_order(state):
    order = await order_service.query_order(state["order_id"], state["user_id"])
    if not order:
        return {"stage": END, "final": True, "message": "订单不存在或不属于您的账号"}
    return {
        "order": _order_to_dict(order),
        "stage": "check_eligibility",
        # 观测层外显业务动作（契约 tool_call），由 orchestrator 统一 emit
        "tool_calls": [{
            "name": "query_order",
            "args": {"order_id": state["order_id"]},
            "result": {"status": order.status, "total_amount": order.total_amount},
            "status": "success",
        }],
    }


async def _check_eligibility(state):
    order = _to_order(state)
    elig = await return_service.check_eligibility(order, state["user_id"])
    if not elig["eligible"]:
        return {"stage": END, "final": True, "message": elig["reason"]}
    # 部分退货：用户指定只退部分商品时，按商品名包含匹配筛子集（整件粒度，不支持部分数量）。
    # 指定商品均不可退时明确拒绝（不回退"退全部"，避免退错），提示可退清单让用户重新发起。
    requested = state.get("return_items") or []
    raw_items = elig.get("items", [])
    if requested:
        selected = [
            it for it in raw_items
            if any(req.strip().lower() in it.name.lower() for req in requested)
        ]
        if not selected:
            names = "、".join(it.name for it in raw_items)
            return {"stage": END, "final": True,
                    "message": f"您指定的商品不支持退货。该订单可退商品：{names}，请重新发起退货并指定正确的商品"}
        refund_amount = round(sum(it.price * it.quantity for it in selected), 2)
        raw_items = selected
    else:
        refund_amount = elig.get("refund_amount", 0)
    # items 转 dict 列表存入 state（OrderItem 对象不可 JSON 序列化）
    elig_saved = {
        "eligible": True,
        "refund_amount": refund_amount,
        "items": [
            {"item_id": i.item_id, "name": i.name, "price": i.price, "quantity": i.quantity}
            for i in raw_items
        ],
    }
    return {"eligibility": elig_saved, "stage": "collect_reason"}


async def _collect_reason(state):
    if state.get("reason"):
        return {"stage": "confirm", "awaiting": None}
    reason = (state.get("user_input") or "").strip()
    if state.get("awaiting") == "reason":
        if _is_deny(reason):
            return {"stage": END, "final": True, "message": "已取消退货操作"}
        if not reason:
            return {"stage": "collect_reason", "awaiting": "reason", "message": "请问退货原因是什么？例如：质量问题、不想要了、尺码不合适"}
        return {"reason": reason, "awaiting": None, "stage": "confirm"}
    return {"stage": "collect_reason", "awaiting": "reason", "message": "请问退货原因是什么？例如：质量问题、不想要了、尺码不合适"}


async def _confirm(state):
    ui = (state.get("user_input") or "").strip()
    if state.get("awaiting") == "confirm":
        if _is_confirm(ui):
            return {"confirmed": True, "awaiting": None, "stage": "execute"}
        if _is_deny(ui):
            # 显式清 awaiting，避免 LangGraph 浅合并残留旧值被上层误判仍在确认中
            return {"stage": END, "final": True, "awaiting": None, "message": "已取消退货操作"}
    elig = state["eligibility"]
    items_desc = "、".join(f"{i['name']}×{i['quantity']}" for i in elig["items"])
    return {
        "stage": "confirm",
        "awaiting": "confirm",
        "message": f"即将为订单 {state['order_id']} 申请退货：{items_desc}，预计退款 ¥{elig['refund_amount']}。回复「确认」提交？",
    }


async def _execute(state):
    order = _to_order(state)
    elig = state["eligibility"]
    item_ids = [i["item_id"] for i in elig["items"]]
    result = await return_service.create_return(
        order, state["user_id"], item_ids, state.get("reason", ""), state["session_id"]
    )
    return {
        "result": {
            "success": result.success,
            "status": result.status,
            "return_id": result.return_id,
            "refund_amount": result.refund_amount,
            "message": result.message,
        },
        "stage": "notify",
        # 观测层外显业务动作（契约 tool_call），由 orchestrator 统一 emit
        "tool_calls": [{
            "name": "create_return",
            "args": {
                "order_id": order.order_id,
                "item_ids": item_ids,
                "reason": state.get("reason", ""),
            },
            "result": {
                "success": result.success,
                "return_id": result.return_id,
                "refund_amount": result.refund_amount,
            },
            "status": "success" if result.success else "error",
        }],
    }


async def _notify(state):
    r = state["result"]
    if r["success"]:
        msg = f"退货单 {r['return_id']} 已创建，退款 ¥{r['refund_amount']} 将在 1-3 个工作日内原路返回。"
    else:
        msg = r["message"]
    return {"stage": END, "final": True, "message": msg}


class ReturnFlow(BaseStateMachine):
    STATE_TYPE = ReturnState
    NODES = {
        "collect_order_id": _collect_order_id,
        "verify_order": _verify_order,
        "check_eligibility": _check_eligibility,
        "collect_reason": _collect_reason,
        "confirm": _confirm,
        "execute": _execute,
        "notify": _notify,
    }
