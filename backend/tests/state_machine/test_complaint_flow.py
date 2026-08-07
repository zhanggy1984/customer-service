"""投诉状态机单元测试（mock reasoner 严重性评估）。"""
import pytest

from app.agent.state_machine import complaint_flow
from app.agent.state_machine.complaint_flow import ComplaintFlow
from app.services import complaint_service
from app.services.models import ComplaintResult


@pytest.mark.asyncio
async def test_complaint_flow(monkeypatch):
    async def fake_create(user_id, order_id, complaint_type, description, severity, session_id):
        return ComplaintResult(success=True, ticket_id="CT-1", severity=severity, message="ok")

    async def fake_assess(desc):
        return "HIGH"

    monkeypatch.setattr(complaint_service, "create_complaint", fake_create)
    monkeypatch.setattr(complaint_flow, "_assess_severity", fake_assess)

    flow = ComplaintFlow()
    state = {"user_id": 1, "session_id": "s", "stage": "collect_complaint_type"}
    state = await flow.step(state, "物流太慢了")
    assert state.get("complaint_type") == "物流问题"

    # step 循环推进：描述 → severity(reasoner) → execute → notify 一气呵成到终态
    state = await flow.step(state, "物流太慢了")
    assert state.get("severity") == "HIGH"
    assert state.get("final") is True
    assert "CT-1" in state.get("message", "")


@pytest.mark.asyncio
async def test_complaint_type_inference():
    from app.agent.state_machine.complaint_flow import _guess_type
    assert _guess_type("商品破损了") == "商品质量"
    assert _guess_type("快递太慢") == "物流问题"
    assert _guess_type("客服态度差") == "服务态度"
    assert _guess_type("随便说说") == "其他"
