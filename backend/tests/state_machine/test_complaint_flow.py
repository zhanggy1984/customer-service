"""投诉状态机单元测试（mock chat 严重性评估）。"""
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


@pytest.mark.asyncio
async def test_assess_severity_uses_chat_model(monkeypatch):
    """优化②：severity 评估走 deepseek-chat（非 reasoner），timeout 用 chat 档。"""
    from app.config import settings
    captured = {}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, **kwargs):
        captured["model"] = model
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": '{"severity":"HIGH"}'}}], "usage": None}

    monkeypatch.setattr(complaint_flow.llm_gateway, "chat", fake_chat)
    sev = await complaint_flow._assess_severity("商品质量极差，手机屏幕碎裂划伤手指")
    assert sev == "HIGH"
    assert captured["model"] == settings.deepseek_model_chat
    assert captured["model"] != settings.deepseek_model_reasoner  # 守护：不回归 reasoner
    assert captured["timeout"] == settings.deepseek_timeout_chat


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,expected", [
    ('{"severity":"HIGH"}', "HIGH"),
    ('{"severity":"MEDIUM"}', "MEDIUM"),
    ('{"severity":"LOW"}', "LOW"),
])
async def test_assess_severity_levels(monkeypatch, payload, expected):
    """三档 severity 透出。"""
    async def fake_chat(messages, model=None, timeout=None, temperature=None, **kwargs):
        return {"choices": [{"message": {"content": payload}}], "usage": None}
    monkeypatch.setattr(complaint_flow.llm_gateway, "chat", fake_chat)
    assert await complaint_flow._assess_severity("test") == expected


@pytest.mark.asyncio
async def test_assess_severity_invalid_falls_back_medium(monkeypatch):
    """非法 severity 值 → 降级 MEDIUM。"""
    async def fake_chat(messages, model=None, timeout=None, temperature=None, **kwargs):
        return {"choices": [{"message": {"content": '{"severity":"URGENT"}'}}], "usage": None}
    monkeypatch.setattr(complaint_flow.llm_gateway, "chat", fake_chat)
    assert await complaint_flow._assess_severity("test") == "MEDIUM"


@pytest.mark.asyncio
async def test_assess_severity_exception_falls_back_medium(monkeypatch):
    """LLM 异常 → 降级 MEDIUM。"""
    async def fake_chat(messages, model=None, timeout=None, temperature=None, **kwargs):
        raise RuntimeError("upstream down")
    monkeypatch.setattr(complaint_flow.llm_gateway, "chat", fake_chat)
    assert await complaint_flow._assess_severity("test") == "MEDIUM"
