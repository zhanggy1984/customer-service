"""Redis 分布式锁单测（手写 FakeRedis，不依赖真实 Redis）。

覆盖：同 sid 并发串行化 / 释放清 key / token 防误删 / 等待超时(429) /
Redis 不可用 fail-fast(503) / 看门狗续期 / SSE 断连取消路径释放。
"""
import asyncio

import pytest

from app.config import settings
from app.session import locks as locks_mod
from app.session.locks import (
    RedisSessionLock,
    SessionLockTimeoutError,
    SessionLockUnavailableError,
)

KEY_PREFIX = locks_mod._LOCK_PREFIX


class FakeRedis:
    """最小 SET NX + Lua(del/pexpire 比对 token) 语义 fake。"""

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, int]] = {}  # key -> (value, ttl_ms)
        self.renew_calls = 0

    def holds(self, key: str) -> bool:
        return key in self._data

    async def set(self, key, value, nx=False, px=None, ex=None):
        if nx and key in self._data:
            return None
        ttl = px if px is not None else (ex * 1000 if ex else None)
        self._data[key] = (value, ttl)
        return True

    async def get(self, key):
        return self._data.get(key, (None, None))[0]

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self._data:
                del self._data[k]
                n += 1
        return n

    async def eval(self, lua, numkeys, *args):
        key, token = args[0], args[1]
        if key not in self._data or self._data[key][0] != token:
            return 0
        if "del" in lua:
            del self._data[key]
            return 1
        if "pexpire" in lua:
            self.renew_calls += 1
            return 1
        return 0


@pytest.fixture
def fake_redis(monkeypatch) -> FakeRedis:
    fake = FakeRedis()
    monkeypatch.setattr(locks_mod, "_redis", lambda: fake)
    return fake


async def _holder(fake: FakeRedis, sid: str, hold: float) -> None:
    """持锁 hold 秒后释放。"""
    async with RedisSessionLock(sid):
        assert fake.holds(f"{KEY_PREFIX}{sid}")
        await asyncio.sleep(hold)


@pytest.mark.asyncio
async def test_lock_serializes_concurrent(fake_redis):
    """同 sid 两个协程：后者须等前者释放才进入临界区（串行）。"""
    entered: list[str] = []
    max_in_critical = 0

    async def worker(name: str):
        nonlocal max_in_critical
        async with RedisSessionLock("s1"):
            entered.append(name)
            max_in_critical = max(max_in_critical, len(entered))
            await asyncio.sleep(0.05)
            max_in_critical = max(max_in_critical, len(entered))
            entered.remove(name)

    await asyncio.gather(worker("A"), worker("B"))
    assert max_in_critical == 1  # 任意时刻临界区内至多 1 个协程（串行互斥）
    assert len(entered) == 0  # 都正常退出


@pytest.mark.asyncio
async def test_release_removes_key(fake_redis):
    async with RedisSessionLock("s2"):
        assert fake_redis.holds(f"{KEY_PREFIX}s2")
    assert not fake_redis.holds(f"{KEY_PREFIX}s2")  # 释放后 key 清除


@pytest.mark.asyncio
async def test_token_prevents_other_owner_release(fake_redis):
    """释放 Lua 比对 token：错误 token 不能删当前持锁者的 key。"""
    key = f"{KEY_PREFIX}s3"
    async with RedisSessionLock("s3"):
        assert fake_redis.holds(key)
        # 另一节点/持有者用错误 token 调释放 Lua → 应拒绝（返回 0），key 保留
        res = await fake_redis.eval(locks_mod._RELEASE_LUA, 1, key, "wrong-token")
        assert res == 0
        assert fake_redis.holds(key)
    assert not fake_redis.holds(key)  # 正确 token 的正常释放 → key 清除


@pytest.mark.asyncio
async def test_wait_timeout_raises(fake_redis, monkeypatch):
    """锁被长期占用且等待超时 → 429 语义异常。"""
    monkeypatch.setattr(settings, "session_lock_wait_timeout", 0.2)
    monkeypatch.setattr(settings, "session_lock_poll_interval", 0.02)
    # 预置一把外部持有的锁
    await fake_redis.set(f"{KEY_PREFIX}s4", "other-token", nx=True, px=60000)

    with pytest.raises(SessionLockTimeoutError):
        async with RedisSessionLock("s4"):
            pass


@pytest.mark.asyncio
async def test_unavailable_raises(monkeypatch):
    """锁 Redis 不可用 → fail-fast（503 语义），不静默降级成无锁。"""
    class BrokenRedis:
        async def set(self, *a, **kw):
            raise ConnectionError("redis down")

    monkeypatch.setattr(locks_mod, "_redis", lambda: BrokenRedis())
    with pytest.raises(SessionLockUnavailableError):
        async with RedisSessionLock("s5"):
            pass


@pytest.mark.asyncio
async def test_watchdog_renews(fake_redis, monkeypatch):
    """看门狗持续续期：持锁期间 Lua 续期被调用（防长处理击穿 TTL）。"""
    monkeypatch.setattr(settings, "session_lock_ttl", 0.6)  # 续期间隔 = 0.6/3 = 0.2s
    async with RedisSessionLock("s6"):
        await asyncio.sleep(0.5)
        assert fake_redis.renew_calls >= 1


@pytest.mark.asyncio
async def test_cancel_releases_lock(fake_redis):
    """SSE 断连 task.cancel：取消路径锁仍释放（shield 保证 eval 执行完）。"""
    key = f"{KEY_PREFIX}s7"

    async def holder():
        async with RedisSessionLock("s7"):
            await asyncio.sleep(10)  # 持锁挂起，等待取消

    task = asyncio.create_task(holder())
    for _ in range(100):  # 等锁拿到
        if fake_redis.holds(key):
            break
        await asyncio.sleep(0.01)
    assert fake_redis.holds(key)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)  # 让 shield 内的释放 eval 后台执行完
    assert not fake_redis.holds(key)  # 取消路径也释放
