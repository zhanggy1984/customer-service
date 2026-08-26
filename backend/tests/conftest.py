"""pytest 配置：集成测试依赖运行中的服务，不可用时跳过。"""
import asyncio

import httpx
import pytest
import redis.asyncio as aioredis

# 共享熔断 key 清理：本会话是否探测过 Redis 可达。本机无 Redis 时首个测试连接失败
# 即置 False（后续跳过，避免每测 1s 连接超时）；容器内首测成功置 True（每测清理 ~ms）。
_shared_breaker_redis_ok: bool | None = None


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
def _fake_session_lock_redis(monkeypatch):
    """单测无真实 Redis：分布式锁退化本地即刻成功（生产走真实 Redis）。

    routes 层直接调路由函数（send_message/delete_session）会拿锁，无 Redis 时
    fail-fast 503 会打断会话逻辑测试。锁自身的互斥/超时/取消语义由
    test_session_lock.py 用精确 FakeRedis 单独覆盖，这里只保证"能拿到锁"。
    """
    from app.session import locks as locks_mod

    class _FakeRedis:
        async def set(self, *a, **kw):
            return True

        async def eval(self, *a, **kw):
            return 1

        async def get(self, *a, **kw):
            return None

        async def delete(self, *a, **kw):
            return 1

    monkeypatch.setattr(locks_mod, "_redis", lambda: _FakeRedis())


@pytest.fixture(autouse=True)
def _reset_module_fault_state():
    """reset 模块级熔断/冷却状态（挑战 2/3）：防跨测试污染。

    retry._breaker（DB 熔断）与 orchestrator._kb_fault_*（检索故障冷却）均为
    进程内存态，测试内触发后会残留下一次调用（如熔断 open 使后续测试的
    query_order 快速失败），必须每个测试前复位。
    """
    import app.agent.orchestrator as orch
    import app.infrastructure.cooldown as cd
    import app.services.retry as retry_mod

    retry_mod._breaker["failures"] = 0
    retry_mod._breaker["open_until"] = 0.0
    orch._kb_fault_streak = 0
    orch._kb_fault_cooldown_until = 0.0
    # RedisCooldown：恢复 Redis 尝试（消除上测试的连接冷却残留）并重建惰性连接
    cd._last_fail = 0.0
    cd._client = None
    # 容器内真实 Redis：清前测广播的共享熔断信号（cs:cb:*:open，TTL 30s 残留会污染
    # 后续测试，见 _clear_shared_breaker_keys）；本机无 Redis 时首测连接失败即进入冷却跳过
    _clear_shared_breaker_keys()
    yield


def _clear_shared_breaker_keys() -> None:
    """容器内真实 Redis 时，前测广播的熔断信号（cs:cb:*:open，TTL 30s）残留会污染后续
    测试：_breaker_open() 命中共享信号 → 非相关测试被快速拒绝（如 test_services 的
    db_circuit_rejected）。best-effort 删除；本机无 Redis 时首个测试连接失败即置
    _shared_breaker_redis_ok=False，本会话后续跳过（避免每测 1s 连接超时）。
    """
    global _shared_breaker_redis_ok
    if _shared_breaker_redis_ok is False:
        return

    from app.config import settings
    from app.infrastructure import cooldown as cd

    async def _clear():
        c = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        try:
            async for key in c.scan_iter("cs:cb:*"):
                await c.delete(key)
        finally:
            await c.aclose()

    try:
        asyncio.run(_clear())
        _shared_breaker_redis_ok = True
    except Exception:
        cd._mark_fail()
        _shared_breaker_redis_ok = False


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
