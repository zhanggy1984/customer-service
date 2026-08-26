"""会话数据 TTL 清理（回收 MySQL 存储）。

- conversation_history / tool_call_log 各按 created_at 独立清理：超保留期即删。
  created_at 语义分别为「最后活跃」/「判定时刻」，保留期同为 session_retention_days。
- 判据全部落在 MySQL 侧：NOW() 与 created_at 的 CURRENT_TIMESTAMP 同会话时区基准，
  Python 不生成 cutoff（避免 aware/naive 混比及时区错位）。
- 分批 DELETE ... LIMIT 循环：控制单次事务行数，避免长事务锁表；批间 sleep 让步。
- 惰性删除（get_session 超期即回收）在 manager.py，本类只管定时全量 sweep。
"""
import asyncio
import logging

from app.config import settings
from app.infrastructure.mysql import mysql_pool

logger = logging.getLogger(__name__)

# 批间让步：给并发写路径让出连接与行锁，防连续大事务阻塞 save
_BATCH_YIELD_SECONDS = 0.05


class SessionCleaner:
    """定时清理 conversation_history / tool_call_log 超期行（仿 deepseek_keypool 启停范式）。"""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """启动后台清理任务（需在事件循环内调用，由 lifespan 触发）。"""
        if self._task is None:
            self._task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(settings.session_cleanup_interval_seconds)
            try:
                total = await self.sweep()
                if total:
                    logger.info("event=session_cleanup_summary", extra={"total": total})
            except Exception as exc:
                # 清理失败不崩应用，下个周期重试
                logger.error("event=session_cleanup_error", extra={"error": str(exc)})

    async def sweep(self) -> int:
        """删除两表超期行，返回删除总行数。分批续删直到单批不足 batch_size。"""
        total = 0
        for table in ("conversation_history", "tool_call_log"):
            while True:
                n = await mysql_pool.execute(
                    f"DELETE FROM {table} WHERE created_at < NOW() - INTERVAL %s DAY LIMIT %s",
                    (settings.session_retention_days, settings.session_cleanup_batch_size),
                )
                total += n
                if n < settings.session_cleanup_batch_size:
                    break
                await asyncio.sleep(_BATCH_YIELD_SECONDS)
        return total


session_cleaner = SessionCleaner()
