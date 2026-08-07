"""DeepSeek Key 池：RPM 滑动窗口 + healthy/cooling 两态 + 后台过期清理。

- KeyState: 每 Key 的 RPM 追踪（60s 滑动窗口），asyncio.Lock 保护
- KeyPool: select_key 选 RPM 最低的 healthy Key；all_cooling 熔断判断
"""
import asyncio
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class KeyState:
    index: int
    api_key: str
    rpm_limit: int = 200
    _timestamps: deque = field(default_factory=deque)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    status: str = "healthy"      # healthy | cooling
    cooldown_until: float = 0.0
    used_count: int = 0          # 累计使用次数（便于验证分布）

    async def is_available(self) -> bool:
        async with self._lock:
            self._drop_expired()
            if self.status == "cooling":
                if time.time() >= self.cooldown_until:
                    self.status = "healthy"
                else:
                    return False
            return len(self._timestamps) < self.rpm_limit * 0.9

    async def get_rpm(self) -> int:
        async with self._lock:
            self._drop_expired()
            return len(self._timestamps)

    async def record_request(self) -> None:
        async with self._lock:
            self._timestamps.append(time.time())
            self.used_count += 1

    async def mark_rate_limited(self, retry_after: int) -> None:
        async with self._lock:
            self.status = "cooling"
            self.cooldown_until = time.time() + retry_after
            self._timestamps.clear()

    def _drop_expired(self) -> None:  # 调用方必须持锁
        cutoff = time.time() - 60
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


class KeyPool:
    def __init__(self, keys: list[str], rpm_limit: int = 200) -> None:
        self._keys = [
            KeyState(index=i, api_key=k, rpm_limit=rpm_limit) for i, k in enumerate(keys)
        ]
        self._cleanup_task: asyncio.Task | None = None

    def start_cleanup(self) -> None:
        """启动后台 30s 清理任务（需在事件循环内调用，由 lifespan init 触发）。"""
        if self._cleanup_task is None and self._keys:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop_cleanup(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def select_key(self) -> KeyState | None:
        """选 healthy 中 RPM 最低的 Key；无可用返回 None。"""
        best: KeyState | None = None
        best_rpm = float("inf")
        for k in self._keys:
            if await k.is_available():
                rpm = await k.get_rpm()
                if rpm < best_rpm:
                    best, best_rpm = k, rpm
        return best

    async def all_cooling(self) -> bool:
        for k in self._keys:
            if await k.is_available():
                return False
        return True

    def healthy_count(self) -> int:
        return sum(1 for k in self._keys if k.status == "healthy")

    def stats(self) -> dict:
        return {
            "healthy": sum(1 for k in self._keys if k.status == "healthy"),
            "cooling": sum(1 for k in self._keys if k.status == "cooling"),
            "keys": len(self._keys),
        }

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            for k in self._keys:
                await k.get_rpm()  # 触发过期时间戳清理
