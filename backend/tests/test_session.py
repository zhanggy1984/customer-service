"""StorageRouter 单元测试：Redis 故障 → 自动切 MySQL。"""
import pytest

from app.session import storage_router as sr_mod
from app.session.models import Session
from app.session.storage_router import StorageRouter


class FakeRedisDown:
    async def get(self, key):
        raise ConnectionError("redis down")

    async def set(self, *a, **kw):
        raise ConnectionError("redis down")

    async def ping(self):
        raise ConnectionError("redis down")

    async def expire(self, *a, **kw):
        return None

    async def delete(self, *a, **kw):
        return None


class FakeMySQL:
    async def execute(self, sql, params=None):
        return 1

    async def fetchone(self, sql, params=None):
        if "conversation_history" in sql:
            return {"user_id": 1, "intent": None, "messages": "[]", "agent_state": "{}"}
        return None


@pytest.mark.asyncio
async def test_router_fallback_to_mysql(monkeypatch):
    monkeypatch.setattr(sr_mod, "mysql_pool", FakeMySQL())
    router = StorageRouter(FakeRedisDown(), ttl=3600)
    await router.start()
    try:
        session = Session(session_id="s1", user_id=1)
        await router.save(session)
        assert router._mode == "mysql_fallback"  # Redis 写失败 → 切换

        loaded = await router.load("s1")
        assert loaded is not None
        assert loaded.user_id == 1
    finally:
        await router.stop()


@pytest.mark.asyncio
async def test_router_normal_mode_redis(monkeypatch):
    class FakeRedisOK:
        async def get(self, key):
            return None  # Redis 有但未命中

        async def set(self, *a, **kw):
            return None

        async def ping(self):
            return True

        async def expire(self, *a, **kw):
            return None

    monkeypatch.setattr(sr_mod, "mysql_pool", FakeMySQL())
    router = StorageRouter(FakeRedisOK(), ttl=3600)
    await router.start()
    try:
        assert router._mode == "redis"  # 初始模式
        session = Session(session_id="s2", user_id=2)
        await router.save(session)
        assert router._mode == "redis"  # Redis 正常，不切换
    finally:
        await router.stop()


@pytest.mark.asyncio
async def test_save_mysql_with_datetime_in_agent_state(monkeypatch):
    """agent_state 含 datetime（状态机还原订单 delivered_at）时，MySQL 兜底写入不应抛异常。

    回归：场景 L 修复引入 datetime 后，json.dumps 抛 TypeError 使 MySQL 会话兜底静默失效。
    """
    from datetime import datetime

    calls: list[tuple] = []

    class FakeMySQL:
        async def execute(self, sql, params=None):
            calls.append((sql, params))
            return 1

    class FakeRedisOK:
        async def get(self, key):
            return None

        async def set(self, *a, **kw):
            return None

        async def ping(self):
            return True

        async def expire(self, *a, **kw):
            return None

    monkeypatch.setattr(sr_mod, "mysql_pool", FakeMySQL())
    router = StorageRouter(FakeRedisOK(), ttl=3600)
    session = Session(
        session_id="s3",
        user_id=1,
        agent_state={"delivered_at": datetime(2026, 8, 3, 15, 0)},
    )
    await router.save(session)  # 不应抛异常

    inserted = [p for s, p in calls if s.startswith("INSERT INTO conversation_history")]
    assert inserted, "应写入 MySQL 会话快照"
    # agent_state_payload（第 5 个参数）中 datetime 已用 default=str 兜底转字符串
    # （str(datetime) 输出空格分隔，如 "2026-08-03 15:00:00"）
    assert '"2026-08-03 15:00:00"' in inserted[0][4]
