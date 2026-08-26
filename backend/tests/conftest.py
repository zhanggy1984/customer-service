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


@pytest.fixture(autouse=True)
def _reset_module_fault_state():
    """reset 模块级熔断/冷却状态（挑战 2/3）：防跨测试污染。

    retry._breaker（DB 熔断）与 orchestrator._kb_fault_*（检索故障冷却）均为
    进程内存态，测试内触发后会残留下一次调用（如熔断 open 使后续测试的
    query_order 快速失败），必须每个测试前复位。
    """
    import app.agent.orchestrator as orch
    import app.services.retry as retry_mod

    retry_mod._breaker["failures"] = 0
    retry_mod._breaker["open_until"] = 0.0
    orch._kb_fault_streak = 0
    orch._kb_fault_cooldown_until = 0.0
    yield


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
