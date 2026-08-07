"""并发会话锁：同一 session 的并发请求串行化。

防止两个请求同时读取同一 session 状态并各自推进，导致状态机覆盖。
锁在事件循环内为每个 session_id 懒创建。
"""
import asyncio


class SessionLocks:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def get(self, sid: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(sid)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[sid] = lock
            return lock


session_locks = SessionLocks()
