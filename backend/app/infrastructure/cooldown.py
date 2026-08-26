"""熔断/冷却的 Redis 共享信号（多节点最小版）。

设计：计数留本地（各节点独立观测连续失败），冷却期广播到 Redis。
- 任一节点达阈值触发冷却 → SET cs:cb:{name}:open 1 EX cooldown（广播"已熔断"）
- 判断：本地冷却中 OR Redis open key 存在 → 降级（他节点不必再打失败调用）
- 成功 → DEL open key

Redis 不可用 → 静默退化本地（熔断是保护性状态，不允许 Redis 故障拖慢主链路）。
首个失败进入 _DISABLE_COOLDOWN 冷却，避免测试/故障环境反复连接超时（同 turn_cache）。
"""
import time

import redis.asyncio as aioredis

from app.config import settings

_DISABLE_COOLDOWN = 60.0  # Redis 失败后本进程内不再尝试连接的时长（秒）
_PREFIX = "cs:cb:"

_client = None
_last_fail = 0.0  # 最近一次 Redis 失败时刻（monotonic），冷却期内不再尝试


def _redis():
    """惰性 Redis 客户端（import 不建连，同 retriever/turn_cache 风格）。"""
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
    return _client


def _disabled() -> bool:
    """Redis 连接冷却期：最近失败后 _DISABLE_COOLDOWN 秒内纯本地判断。"""
    return time.monotonic() - _last_fail < _DISABLE_COOLDOWN


def _mark_fail() -> None:
    global _last_fail
    _last_fail = time.monotonic()


class RedisCooldown:
    """单个熔断器的共享冷却信号（实例独立 key，模块级共享 Redis 连接）。

    name 取值约定：llm / kb / db（对应三个熔断/冷却消费点）。
    """

    def __init__(self, name: str, cooldown: float) -> None:
        self._key = f"{_PREFIX}{name}:open"
        self._cooldown = cooldown

    async def is_open(self) -> bool:
        """Redis 侧冷却中？（本地冷却由调用方各自维护，这里只管共享信号）"""
        if _disabled():
            return False
        try:
            return bool(await _redis().exists(self._key))
        except Exception:
            _mark_fail()
            return False

    async def open(self) -> None:
        """广播冷却开始（SET EX cooldown，覆盖旧值）。"""
        if _disabled():
            return
        try:
            await _redis().set(self._key, "1", ex=self._cooldown)
        except Exception:
            _mark_fail()

    async def close(self) -> None:
        """成功重置 → 清除共享冷却信号。"""
        if _disabled():
            return
        try:
            await _redis().delete(self._key)
        except Exception:
            _mark_fail()
