"""StorageRouter 单元测试：Redis 故障 → 自动切 MySQL。"""
import pytest

from app.session import storage_router as sr_mod
from app.session.models import Message, Session
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


@pytest.mark.asyncio
async def test_save_mysql_with_datetime_in_messages(monkeypatch):
    """messages 含 datetime（ts）时，MySQL 兜底写入不应抛异常。

    回归：model_dump() 默认保留 datetime 对象，裸 json.dumps 抛 TypeError；
    修复为 mode="json"，与 Redis 的 model_dump_json() 口径一致输出 ISO 字符串。
    """
    from datetime import datetime, timezone

    from app.session.models import Message

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
    ts = datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc)
    session = Session(
        session_id="s4",
        user_id=1,
        messages=[Message(role="user", content="hi", ts=ts)],
    )
    await router.save(session)  # 不应抛异常

    inserted = [p for s, p in calls if s.startswith("INSERT INTO conversation_history")]
    assert inserted, "应写入 MySQL 会话快照"
    # messages（第 4 个参数）中 ts 已序列化为 ISO 字符串，而非 datetime 对象
    assert "2026-08-03T15:00:00" in inserted[0][3]


# ---------- Session.trim 消息体截断 ----------

def _msgs(n: int, start_with_user: bool = True) -> list[Message]:
    """构造 n 条交替角色消息，index 与 content 对应（m0 开始）。"""
    return [
        Message(role="user" if (i % 2 == 0) == start_with_user else "assistant", content=f"m{i}")
        for i in range(n)
    ]


def test_trim_within_limit_untouched():
    session = Session(session_id="t1", user_id=1, messages=_msgs(5))
    session.trim(10)
    assert len(session.messages) == 5


def test_trim_exactly_limit_untouched():
    session = Session(session_id="t2", user_id=1, messages=_msgs(10))
    session.trim(10)
    assert len(session.messages) == 10


def test_trim_keeps_first_user_and_recent():
    """超过上限：保留首条 user 消息（标题锚点）+ 最近 N-1 条，中间丢弃。"""
    session = Session(session_id="t3", user_id=1, messages=_msgs(50))
    session.trim(10)
    assert len(session.messages) == 10
    assert session.messages[0].content == "m0"    # 首条 user 保留
    assert session.messages[0].role == "user"
    assert session.messages[-1].content == "m49"  # 最近一条保留
    assert session.messages[1].content == "m41"   # 跳过中间，接最近一段


def test_trim_first_not_user_keeps_only_recent():
    """首条非 user（assistant 开头）：无标题锚点可保，只取最近 N 条。"""
    msgs = [Message(role="assistant", content="welcome")] + _msgs(30)  # 31 条，尾部 10 条是 m20..m29
    session = Session(session_id="t4", user_id=1, messages=msgs)
    session.trim(10)
    assert len(session.messages) == 10
    assert session.messages[0].content == "m20"
    assert session.messages[-1].content == "m29"


def test_trim_empty_untouched():
    session = Session(session_id="t5", user_id=1)
    session.trim(10)
    assert session.messages == []


def test_trim_max_one_keeps_only_anchor():
    """max=1 且首条是 user：keep=0 时只保留首条锚点，不重复、不残留其他。"""
    session = Session(session_id="t6", user_id=1, messages=_msgs(5))
    session.trim(1)
    assert len(session.messages) == 1
    assert session.messages[0].content == "m0"
    assert session.messages[0].role == "user"
