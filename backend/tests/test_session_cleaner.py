"""会话 TTL 清理单测（conversation_history / tool_call_log 超期回收）。

核心逻辑：sweep 分批删除、is_expired 超期判定、get_session 惰性过期回收、
delete 级联删 tool_call_log。不连真实 MySQL：mock mysql_pool / router。
"""
import pytest

from app.infrastructure.mysql import mysql_pool
from app.session.cleaner import session_cleaner
from app.session.manager import session_manager
from app.session.models import Session
from app.session.storage_router import StorageRouter


# ---------- sweep ----------


@pytest.mark.asyncio
async def test_sweep_deletes_expired_in_batches(monkeypatch):
    """两表按 created_at < NOW()-INTERVAL 分批删除（满批续删），返回删除总行数。"""
    calls: list[tuple] = []
    # conversation_history: 500/500/3（两满批 + 尾批）；tool_call_log: 500/2（一满批 + 尾批）
    counts = iter([500, 500, 3, 500, 2])

    async def fake_execute(sql, params):
        calls.append((sql, params))
        return next(counts)

    monkeypatch.setattr(mysql_pool, "execute", fake_execute)
    total = await session_cleaner.sweep()

    assert total == 1505
    assert len(calls) == 5
    # 两表 SQL 均为条件删除 + LIMIT 分批，参数为保留天数/批量
    for sql, params in calls:
        assert "DELETE FROM" in sql and "created_at < NOW() - INTERVAL %s DAY" in sql and "LIMIT %s" in sql
        assert params == (30, 500)
    # 两表各有一次满批续删 + 尾批停止（< batch_size 退出）
    assert calls[0][0].endswith("conversation_history WHERE created_at < NOW() - INTERVAL %s DAY LIMIT %s")


@pytest.mark.asyncio
async def test_sweep_no_expired_rows(monkeypatch):
    """无超期行（单次即不足批量）→ 每表一次 DELETE，返回 0。"""
    calls: list[tuple] = []

    async def fake_execute(sql, params):
        calls.append((sql, params))
        return 0

    monkeypatch.setattr(mysql_pool, "execute", fake_execute)
    assert await session_cleaner.sweep() == 0
    assert len(calls) == 2  # 两表各一次，n=0 < 500 直接退出


# ---------- is_expired ----------


@pytest.mark.asyncio
async def test_is_expired_true_when_over_retention(monkeypatch):
    """存在 created_at < cutoff 的行 → 会话超期。判据在 MySQL 侧 NOW()（与 CURRENT_TIMESTAMP 同基准）。"""
    captured = {}

    async def fake_fetchone(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return {"created_at": "2020-01-01 00:00:00"}

    monkeypatch.setattr(mysql_pool, "fetchone", fake_fetchone)
    router = StorageRouter(None, 3600)
    assert await router.is_expired("sess-1", 30) is True
    assert "created_at < NOW() - INTERVAL %s DAY" in captured["sql"]
    assert captured["params"] == ("sess-1", 30)


@pytest.mark.asyncio
async def test_is_expired_false_when_active(monkeypatch):
    """无超期行 → 会话活跃。"""
    async def fake_fetchone(sql, params):
        return None

    monkeypatch.setattr(mysql_pool, "fetchone", fake_fetchone)
    router = StorageRouter(None, 3600)
    assert await router.is_expired("sess-1", 30) is False


# 边界语义：created_at 恰等于 cutoff 不删（`<` 严格）已由
# test_is_expired_true_when_over_retention 的 SQL 断言覆盖（`<` 而非 `<=`）。


# ---------- get_session 惰性过期 ----------


class _FakeRouter:
    def __init__(self, expired: bool) -> None:
        self._expired = expired
        self._loaded = Session(session_id="sess-1", user_id=1)
        self.deleted: list[str] = []

    async def load(self, sid: str):
        return self._loaded

    async def is_expired(self, sid: str, days: int) -> bool:
        return self._expired

    async def delete(self, sid: str) -> None:
        self.deleted.append(sid)


@pytest.mark.asyncio
async def test_get_session_lazy_expires(monkeypatch):
    """惰性过期：走 MySQL 兜底恢复时判超期 → 物理回收（delete）+ 返回 None。"""
    fake = _FakeRouter(expired=True)
    monkeypatch.setattr(session_manager, "_router", fake)
    assert await session_manager.get_session("sess-1") is None
    assert fake.deleted == ["sess-1"]  # 触发回收


@pytest.mark.asyncio
async def test_get_session_keeps_active(monkeypatch):
    """未超期 → 正常返回会话，不触发回收。"""
    fake = _FakeRouter(expired=False)
    monkeypatch.setattr(session_manager, "_router", fake)
    session = await session_manager.get_session("sess-1")
    assert session is not None and session.session_id == "sess-1"
    assert fake.deleted == []


# ---------- delete 级联删 tool_call_log ----------


class _FakeRedis:
    """async delete 记录被删 key（类内定义避免闭包被绑定成实例方法多传 self）。"""

    def __init__(self, deleted_keys: list[str]) -> None:
        self._deleted_keys = deleted_keys

    async def delete(self, key: str) -> None:
        self._deleted_keys.append(key)


@pytest.mark.asyncio
async def test_delete_cascades_tool_call_log(monkeypatch):
    """delete(sid) 级联回收：Redis + conversation_history + tool_call_log（既有缺口修复）。"""
    deleted_keys: list[str] = []
    executed: list[str] = []

    async def fake_execute(sql, params):
        executed.append(sql)

    router = StorageRouter(None, 3600)
    router._redis = _FakeRedis(deleted_keys)
    monkeypatch.setattr(mysql_pool, "execute", fake_execute)
    await router.delete("sess-1")
    assert deleted_keys == ["session:sess-1"]
    assert executed == [
        "DELETE FROM conversation_history WHERE session_id=%s",
        "DELETE FROM tool_call_log WHERE session_id=%s",
    ]
