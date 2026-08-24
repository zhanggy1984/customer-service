"""状态机契约单测：退款/退货/投诉流程透出 tool_calls 观测事件（契约 §5.1）。

节点返回 state["tool_calls"]/reasoning，由 orchestrator 统一 emit tool_call_event /
reasoning_event。覆盖 verify_order / execute / severity_assess 核心节点的观测动作。
"""
import pytest

from app.agent.state_machine import complaint_flow, refund_flow, return_flow
from app.agent.state_machine.refund_flow import _verify_order as refund_verify
from app.agent.state_machine.return_flow import _verify_order as return_verify
from app.services.models import (
    ComplaintResult,
    OrderInfo,
    OrderItem,
    RefundResult,
    ReturnResult,
)


def _order(status: str = "PAID") -> OrderInfo:
    return OrderInfo(
        order_id="ORD-1", user_id=1, status=status, total_amount=99.9,
        items=[OrderItem(id=1, item_id="SKU-1", name="手机壳", price=99.9,
                         quantity=1, returnable=True)],
    )


def _order_dict(order: OrderInfo) -> dict:
    from app.agent.function_calling.tools.order_tools import _order_to_dict
    return _order_to_dict(order)


# ---------- verify_order ----------

@pytest.mark.asyncio
async def test_refund_verify_emits_query_order_tool_call(monkeypatch):
    """退款流 verify_order：透出 query_order 动作，含订单状态/金额。"""
    async def fake_query(oid, uid):
        return _order("PAID")
    monkeypatch.setattr(refund_flow.order_service, "query_order", fake_query)
    new = await refund_verify({"order_id": "ORD-1", "user_id": 1})
    tc = new.get("tool_calls")
    assert tc and tc[0]["name"] == "query_order"
    assert tc[0]["result"]["status"] == "PAID"
    assert tc[0]["result"]["total_amount"] == 99.9
    assert new["stage"] == "check_refund_eligibility"


@pytest.mark.asyncio
async def test_return_verify_emits_query_order_tool_call(monkeypatch):
    """退货流 verify_order：透出 query_order 动作。"""
    async def fake_query(oid, uid):
        return _order("DELIVERED")
    monkeypatch.setattr(return_flow.order_service, "query_order", fake_query)
    new = await return_verify({"order_id": "ORD-1", "user_id": 1})
    tc = new.get("tool_calls")
    assert tc and tc[0]["name"] == "query_order"
    assert tc[0]["result"]["status"] == "DELIVERED"
    assert new["stage"] == "check_eligibility"


# ---------- execute ----------

@pytest.mark.asyncio
async def test_refund_execute_emits_create_refund(monkeypatch):
    """退款流 execute：透出 create_refund 动作。"""
    async def fake_create(order, uid, reason, sid):
        return RefundResult(success=True, refund_id="RF-1", status="PENDING", amount=99.9)
    monkeypatch.setattr(refund_flow.refund_service, "create_refund", fake_create)
    state = {"order": _order_dict(_order("PAID")), "user_id": 1,
             "reason": "不想要了", "session_id": "s", "eligibility": {"amount": 99.9}}
    new = await refund_flow._execute(state)
    tc = new.get("tool_calls")
    assert tc and tc[0]["name"] == "create_refund"
    assert tc[0]["args"]["order_id"] == "ORD-1"
    assert tc[0]["result"]["refund_id"] == "RF-1"
    assert tc[0]["status"] == "success"


@pytest.mark.asyncio
async def test_return_execute_emits_create_return(monkeypatch):
    """退货流 execute：透出 create_return 动作，含 item_ids。"""
    async def fake_create(order, uid, item_ids, reason, sid):
        return ReturnResult(success=True, return_id="RT-1", status="PENDING", refund_amount=99.9)
    monkeypatch.setattr(return_flow.return_service, "create_return", fake_create)
    state = {
        "order": _order_dict(_order("DELIVERED")), "user_id": 1,
        "eligibility": {"items": [{"item_id": "SKU-1", "name": "手机壳", "price": 99.9, "quantity": 1}]},
        "reason": "不想要了", "session_id": "s",
    }
    new = await return_flow._execute(state)
    tc = new.get("tool_calls")
    assert tc and tc[0]["name"] == "create_return"
    assert tc[0]["args"]["item_ids"] == ["SKU-1"]
    assert tc[0]["result"]["return_id"] == "RT-1"


# ---------- complaint ----------

@pytest.mark.asyncio
async def test_complaint_severity_assess_reasoning(monkeypatch):
    """投诉流 severity_assess：透出 reasoning（严重性评估依据）。"""
    async def fake_assess(desc):
        return "HIGH"
    monkeypatch.setattr(complaint_flow, "_assess_severity", fake_assess)
    new = await complaint_flow._severity_assess({"description": "商品破损"})
    assert new["severity"] == "HIGH"
    assert "HIGH" in new["reasoning"] and "紧急" in new["reasoning"]


@pytest.mark.asyncio
async def test_complaint_execute_emits_create_complaint(monkeypatch):
    """投诉流 execute：透出 create_complaint 动作。"""
    async def fake_create(**kw):
        return ComplaintResult(success=True, ticket_id="T-1", severity="MEDIUM")
    monkeypatch.setattr(complaint_flow.complaint_service, "create_complaint", fake_create)
    state = {"user_id": 1, "session_id": "s", "complaint_type": "商品质量",
             "order_id": "ORD-1", "description": "坏了", "severity": "MEDIUM"}
    new = await complaint_flow._execute(state)
    tc = new.get("tool_calls")
    assert tc and tc[0]["name"] == "create_complaint"
    assert tc[0]["args"]["complaint_type"] == "商品质量"
    assert tc[0]["result"]["ticket_id"] == "T-1"


# ---------- rule_engine 话术 ----------

def test_rule_engine_no_placeholder_hotline():
    """移除假热线后：规则引擎所有话术不再含占位号码（评测 judge 扣分点）。"""
    from app.agent.rule_engine import DEFAULT_REPLY, RULES
    texts = [reply for _, reply in RULES] + [DEFAULT_REPLY]
    for t in texts:
        assert "400-XXX" not in t
        assert "XXX-XXXX" not in t
