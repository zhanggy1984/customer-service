"""仅退款状态机（LangGraph）。

节点链: collect_order_id → verify_order → check_refund_eligibility → collect_reason → confirm → execute → notify
check_refund_eligibility 使用 IRefundService 三级判定（PAID 可退 / SHIPPED 需拒收 / DELIVERED 走退货）。
"""
from typing import TypedDict

from langgraph.graph import END

from app.agent.function_calling.tools.order_tools import _order_to_dict
from app.agent.state_machine.base import BaseStateMachine
from app.agent.state_machine.edges import is_confirm, is_deny, order_from_state
from app.services import order_service, refund_service


class RefundState(TypedDict, total=False):
    user_id: int
    session_id: str
    order_id: str
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
    return {"order": _order_to_dict(order), "stage": "check_refund_eligibility"}


async def _check_refund_eligibility(state):
    order = order_from_state(state)
    elig = await refund_service.check_refund_eligibility(order)
    if not elig["eligible"]:
        return {"stage": END, "final": True, "message": elig["reason"]}
    return {"eligibility": elig, "stage": "collect_reason"}


async def _collect_reason(state):
    if state.get("reason"):
        return {"stage": "confirm", "awaiting": None}
    reason = (state.get("user_input") or "").strip()
    if state.get("awaiting") == "reason":
        if is_deny(reason):
            return {"stage": END, "final": True, "message": "已取消仅退款操作"}
        if not reason:
            return {"stage": "collect_reason", "awaiting": "reason", "message": "请问仅退款原因是什么？例如：未发货不想要了"}
        return {"reason": reason, "awaiting": None, "stage": "confirm"}
    return {"stage": "collect_reason", "awaiting": "reason", "message": "请问仅退款原因是什么？例如：未发货不想要了"}


async def _confirm(state):
    ui = (state.get("user_input") or "").strip()
    if state.get("awaiting") == "confirm":
        if is_confirm(ui):
            return {"confirmed": True, "awaiting": None, "stage": "execute"}
        if is_deny(ui):
            # 显式清 awaiting，避免 LangGraph 浅合并残留旧值被上层误判仍在确认中
            return {"stage": END, "final": True, "awaiting": None, "message": "已取消仅退款操作"}
    elig = state["eligibility"]
    return {
        "stage": "confirm",
        "awaiting": "confirm",
        "message": f"即将为订单 {state['order_id']} 申请仅退款 ¥{elig['amount']}。回复「确认」提交？",
    }


async def _execute(state):
    order = order_from_state(state)
    result = await refund_service.create_refund(
        order, state["user_id"], state.get("reason", ""), state["session_id"]
    )
    return {
        "result": {
            "success": result.success,
            "status": result.status,
            "refund_id": result.refund_id,
            "amount": result.amount,
            "message": result.message,
        },
        "stage": "notify",
    }


async def _notify(state):
    r = state["result"]
    if r["success"]:
        msg = f"仅退款申请已提交（单号 {r['refund_id']}），退款 ¥{r['amount']} 将在 1-3 个工作日内原路退回。"
    else:
        msg = r["message"]
    return {"stage": END, "final": True, "message": msg}


class RefundFlow(BaseStateMachine):
    STATE_TYPE = RefundState
    NODES = {
        "collect_order_id": _collect_order_id,
        "verify_order": _verify_order,
        "check_refund_eligibility": _check_refund_eligibility,
        "collect_reason": _collect_reason,
        "confirm": _confirm,
        "execute": _execute,
        "notify": _notify,
    }
