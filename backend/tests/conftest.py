"""pytest 配置：集成测试依赖运行中的服务，不可用时跳过。"""
import httpx
import pytest


@pytest.fixture(autouse=True)
def _noop_tool_call_log(monkeypatch):
    """单测不落真实 MySQL：护栏判定落库（P5）替换为 no-op。

    agent_loop 里 write_tool_call 是 `from ...tool_call_log import` 绑定的名字，
    须 patch agent_loop 命名空间而非源模块。落库函数自身逻辑由 test_tool_call_log.py 覆盖。
    """
    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr("app.agent.agent_loop.write_tool_call", _noop)


def _service_ready() -> bool:
    try:
        resp = httpx.get("http://localhost:8000/healthz", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


SERVICE_READY = _service_ready()


@pytest.fixture
def service_ready() -> bool:
    return SERVICE_READY
