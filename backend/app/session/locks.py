"""并发会话锁：同一 session 的并发请求串行化（Redis 分布式锁，多节点共享）。

替代原进程内 asyncio.Lock：同 session 并发请求无论落在哪个节点，都通过
Redis SET NX PX 互斥串行化，防止两请求同时读改写导致状态机覆盖。

- 获取：SET cs:lock:session:{sid} token NX PX ttl；失败 → 轮询重试至 wait_timeout
- 释放：Lua 比对 token 原子删除（防误删他人锁）
- 看门狗：持锁期间每 ttl/3 续期（Lua 比对 token），防长处理（LLM 多轮决策）击穿 TTL
- 锁 Redis 不可用 → fail-fast（锁是正确性依赖，静默降级 = 并发错乱）
- 等待超时 → 抛 SessionLockTimeoutError（上层映射 429）

调用方接口不变：`lock = await session_locks.get(sid)` + `async with lock:`。
"""
import asyncio
import os
import socket
import time
import uuid

import redis.asyncio as aioredis

from app.config import settings
from app.infrastructure import metrics

_LOCK_PREFIX = "cs:lock:session:"
# Lua：释放（比对 token，防误删持锁者之外的锁）
_RELEASE_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
else
  return 0
end
"""
# Lua：续期（比对 token，仅持锁者可续）
_RENEW_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("pexpire", KEYS[1], ARGV[2])
else
  return 0
end
"""

_client = None


def _redis() -> aioredis.Redis:
    """惰性 Redis 客户端（锁是正确性依赖，不设过短超时）。"""
    global _client
    if _client is None:
        _client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _client


class SessionLockUnavailableError(Exception):
    """锁 Redis 不可用（fail-fast，上层映射 503）。"""


class SessionLockTimeoutError(Exception):
    """同会话持续被处理，等待超时（上层映射 429）。"""


class RedisSessionLock:
    """单个 session 的 Redis 分布式互斥锁（可 async with）。"""

    def __init__(self, sid: str) -> None:
        self._sid = sid
        self._key = f"{_LOCK_PREFIX}{sid}"
        # token：节点唯一（hostname:pid:uuid），释放/续期 Lua 比对，防误删他人锁
        self._token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        self._ttl_ms = int(settings.session_lock_ttl * 1000)
        self._acquired = False
        self._watchdog: asyncio.Task | None = None

    async def __aenter__(self) -> "RedisSessionLock":
        await self._acquire()
        self._start_watchdog()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # CancelledError（SSE 断连 task.cancel）也会进入这里：释放必须容忍取消
        await self._release()

    async def _acquire(self) -> None:
        deadline = time.monotonic() + settings.session_lock_wait_timeout
        while True:
            try:
                ok = await _redis().set(self._key, self._token, nx=True, px=self._ttl_ms)
            except Exception as exc:
                raise SessionLockUnavailableError("锁服务不可用") from exc
            if ok:
                self._acquired = True
                return
            if time.monotonic() >= deadline:
                metrics.inc("session_lock_timeout")
                raise SessionLockTimeoutError(f"会话 {self._sid} 正在处理中")
            await asyncio.sleep(settings.session_lock_poll_interval)

    async def _release(self) -> None:
        """取消看门狗 + Lua 比对 token 释放。

        shield 包住释放：外部 task 取消（SSE 断连 task.cancel）时，CancelledError
        会在 __aexit__ 的任何 await 点注入，不用 shield 会跳过释放；shield 保证
        eval 在后台执行完（大概率释放成功）。最坏仍失败则锁 TTL 兜底自动过期。
        """
        if self._watchdog is not None:
            self._watchdog.cancel()
            try:
                await self._watchdog
            except asyncio.CancelledError:
                pass
            self._watchdog = None
        if self._acquired:
            self._acquired = False
            try:
                await asyncio.shield(_redis().eval(_RELEASE_LUA, 1, self._key, self._token))
            except (Exception, asyncio.CancelledError):
                pass

    def _start_watchdog(self) -> None:
        self._watchdog = asyncio.create_task(self._watchdog_loop())

    async def _watchdog_loop(self) -> None:
        """每 ttl/3 续期一次；Redis 瞬时抖动不退出，下个周期再试（持续不可用靠 TTL 兜底）。"""
        interval = self._ttl_ms / 1000 / 3
        while True:
            await asyncio.sleep(interval)
            try:
                await _redis().eval(_RENEW_LUA, 1, self._key, self._token, self._ttl_ms)
            except Exception:
                # 一次续期失败不放弃：Redis 抖动恢复后继续续期，防长处理击穿 TTL
                continue


class SessionLocks:
    """锁获取入口：get(sid) 返回该会话的分布式锁实例（互斥态在 Redis，实例每次新建）。"""

    async def get(self, sid: str) -> RedisSessionLock:
        return RedisSessionLock(sid)


session_locks = SessionLocks()
