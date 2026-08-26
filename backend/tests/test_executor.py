"""executor.execute 单测：未知工具 / handler 异常 → 统一错误信封（FC 契约）。"""
import pytest

from app.agent.function_calling import executor


@pytest.mark.asyncio
async def test_execute_unknown_tool():
    """未注册工具 → unknown_tool 信封。"""
    out = await executor.execute("no_such_tool", {}, 1)
    assert out["ok"] is False
    assert out["data"] is None
    assert out["error"]["code"] == "unknown_tool"
    assert "no_such_tool" in out["error"]["message"]


@pytest.mark.asyncio
async def test_execute_handler_error_envelope(monkeypatch):
    """handler 抛异常 → internal_error 信封（不泄露内部错误）。"""
    async def boom(params, user_id, session_id):
        raise RuntimeError("db down")

    monkeypatch.setitem(executor.HANDLERS, "boom_tool", boom)
    out = await executor.execute("boom_tool", {}, 1)
    assert out["ok"] is False
    assert out["error"]["code"] == "internal_error"
    assert "系统出问题" in out["error"]["message"]


@pytest.mark.asyncio
async def test_execute_success_passthrough(monkeypatch):
    """handler 成功 → 透传其信封，executor 不二次包装。"""
    async def ok(params, user_id, session_id):
        return {"ok": True, "data": {"order": {}}, "error": None}

    monkeypatch.setitem(executor.HANDLERS, "ok_tool", ok)
    out = await executor.execute("ok_tool", {}, 1)
    assert out == {"ok": True, "data": {"order": {}}, "error": None}
