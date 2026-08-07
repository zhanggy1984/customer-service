"""意图分类器单元测试（mock LLM，不真实调用 DeepSeek）。"""
import pytest

from app.agent.intent import _chitchat_fallback, _extract_json, classify_intent
from app.infrastructure.deepseek import deepseek_client


def test_extract_json_markdown_wrapped():
    assert _extract_json('```json\n{"intent":"A"}\n```') == '{"intent":"A"}'


def test_extract_json_trailing_text():
    assert _extract_json('以下是结果：{"intent":"A"} 请查收') == '{"intent":"A"}'


@pytest.mark.asyncio
async def test_classify_return_request(monkeypatch):
    async def fake_chat(messages, model=None, timeout=None):
        return {"choices": [{"message": {"content": '{"intent":"RETURN_REQUEST","confidence":0.98,"slots":{},"missing_slots":["order_id"],"summary":"退货"}'}}]}
    monkeypatch.setattr(deepseek_client, "chat", fake_chat)
    r = await classify_intent("我要退货")
    assert r.intent == "RETURN_REQUEST"
    assert r.missing_slots == ["order_id"]


@pytest.mark.asyncio
async def test_classify_chitchat(monkeypatch):
    async def fake_chat(messages, model=None, timeout=None):
        return {"choices": [{"message": {"content": '{"intent":"CHITCHAT","confidence":0.99,"slots":{},"missing_slots":[],"summary":"问候"}'}}]}
    monkeypatch.setattr(deepseek_client, "chat", fake_chat)
    r = await classify_intent("你好")
    assert r.intent == "CHITCHAT"


@pytest.mark.asyncio
async def test_classify_invalid_json_retries_then_fallback(monkeypatch):
    calls = {"n": 0}

    async def fake_chat(messages, model=None, timeout=None):
        calls["n"] += 1
        return {"choices": [{"message": {"content": "这不是JSON"}}]}
    monkeypatch.setattr(deepseek_client, "chat", fake_chat)
    r = await classify_intent("乱码", max_retries=2)
    assert r.intent == "CHITCHAT"  # 兜底
    assert calls["n"] == 2  # 重试耗尽


@pytest.mark.asyncio
async def test_classify_invalid_intent_fallback(monkeypatch):
    async def fake_chat(messages, model=None, timeout=None):
        return {"choices": [{"message": {"content": '{"intent":"UNKNOWN_TYPE","confidence":0.9,"slots":{},"missing_slots":[],"summary":""}'}}]}
    monkeypatch.setattr(deepseek_client, "chat", fake_chat)
    r = await classify_intent("x")
    assert r.intent == "CHITCHAT"


def test_fallback():
    assert _chitchat_fallback().intent == "CHITCHAT"
