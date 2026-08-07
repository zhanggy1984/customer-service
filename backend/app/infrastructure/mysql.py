"""MySQL 异步连接池（asyncmy）。

全局单例 mysql_pool，由 FastAPI lifespan 初始化 / 关闭。
池参数从 .env 读取（MYSQL_POOL_SIZE / MYSQL_MAX_OVERFLOW）。
"""
import logging
from contextlib import asynccontextmanager

import asyncmy
from asyncmy.cursors import DictCursor
from sqlalchemy.engine.url import make_url

from app.config import settings

logger = logging.getLogger(__name__)


class MySQLPool:
    """asyncmy 连接池封装：fetchone / fetchall / execute。"""

    def __init__(self) -> None:
        self._pool = None  # asyncmy.Pool | None

    async def init(self) -> None:
        if self._pool is not None:
            return
        url = make_url(settings.mysql_url)
        # asyncmy 原生池语义为 minsize/maxsize（无 SQLAlchemy 的 max_overflow 概念）
        self._pool = await asyncmy.create_pool(
            host=url.host,
            port=url.port or 3306,
            user=url.username,
            password=url.password,
            database=url.database,
            minsize=5,
            maxsize=settings.mysql_pool_size,
            pool_recycle=3600,  # 连接复用上限 1h，防止服务端空闲断开
            autocommit=True,
            charset="utf8mb4",
        )
        logger.info("event=mysql_pool_init host=%s db=%s", url.host, url.database)

    async def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            logger.info("event=mysql_pool_closed")

    async def fetchone(self, sql: str, params: tuple | list | None = None) -> dict | None:
        async with self._pool.acquire() as conn:
            async with conn.cursor(DictCursor) as cur:
                await cur.execute(sql, params or ())
                return await cur.fetchone()

    async def fetchall(self, sql: str, params: tuple | list | None = None) -> list[dict]:
        async with self._pool.acquire() as conn:
            async with conn.cursor(DictCursor) as cur:
                await cur.execute(sql, params or ())
                rows = await cur.fetchall()
                return list(rows or [])

    async def execute(self, sql: str, params: tuple | list | None = None) -> int:
        """执行写操作，返回影响行数（autocommit=True，无需手动 commit）。"""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params or ())
                return cur.rowcount

    @asynccontextmanager
    async def transaction(self):
        """显式事务（同一连接内多语句原子执行）。

        用法:
            async with mysql_pool.transaction() as run:
                cur = await run("INSERT INTO orders (...) VALUES (...)", (...))
                order_db_id = cur.lastrowid
        正常结束自动 COMMIT，异常自动 ROLLBACK。
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("START TRANSACTION")

            async def run(sql: str, params: tuple | list | None = None):
                async with conn.cursor() as cur:
                    await cur.execute(sql, params or ())
                    return cur

            try:
                yield run
                async with conn.cursor() as cur:
                    await cur.execute("COMMIT")
            except Exception:
                async with conn.cursor() as cur:
                    await cur.execute("ROLLBACK")
                raise


mysql_pool = MySQLPool()
