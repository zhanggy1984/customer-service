"""应用自建表：共享 mysql 不执行 init.sql（原挂容器 init 机制已移除），应用启动时执行建表+种子。

背景：cs 是唯一依赖 mysql 容器 init 脚本建表的 agent（无 alembic/create_all）。
切共享 infra 后 mysql 不跑 init.sql，故由应用 lifespan 执行 backend/sql/init.sql。
建表用 CREATE TABLE IF NOT EXISTS 幂等；种子带 INSERT IGNORE / 空表守卫，重复启动不重复插入。
"""
import logging
from pathlib import Path

from app.infrastructure.mysql import mysql_pool

logger = logging.getLogger(__name__)

# backend/sql/init.sql（Dockerfile COPY . . 进镜像 /app/sql/init.sql）
INIT_SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "init.sql"


def _has_sql(segment: str) -> bool:
    """段内是否含有效 SQL：跳过以 -- 开头的整行注释后仍有内容

    init.sql 每个语句前都带 `-- ---------- xxx ----------` 注释行，
    不能按"段以 -- 开头"丢弃整段，否则建表/种子全被跳过（表将缺失）。
    """
    return any(
        line.strip() and not line.strip().startswith("--")
        for line in segment.splitlines()
    )


async def init_schema() -> None:
    """执行 init.sql（按分号切分为单条语句逐条执行）。"""
    if not INIT_SQL_PATH.exists():
        logger.error("init.sql 不存在: %s，跳过建表（表将缺失）", INIT_SQL_PATH)
        return
    text = INIT_SQL_PATH.read_text(encoding="utf-8")
    statements = [
        s.strip()
        for s in text.split(";")
        if _has_sql(s)
    ]
    for stmt in statements:
        await mysql_pool.execute(stmt)
    await _ensure_knowledge_hash_column()
    logger.info("schema init done（%s 条语句）", len(statements))


async def _ensure_knowledge_hash_column() -> None:
    """存量库迁移：CREATE TABLE IF NOT EXISTS 不会给已存在表加列，
    knowledge_docs.content_hash 缺失时补列（增量跳检的判据，见 kb_store）。

    用 information_schema 检查而非 try/except 吞 ALTER 异常，避免掩盖真实 SQL 错误。
    init_schema 先执行建表语句，此函数运行时表必然存在。
    """
    row = await mysql_pool.fetchone(
        "SELECT COUNT(*) AS c FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'knowledge_docs' "
        "AND COLUMN_NAME = 'content_hash'"
    )
    if (row or {}).get("c", 0) == 0:
        await mysql_pool.execute("ALTER TABLE knowledge_docs ADD COLUMN content_hash CHAR(64) NULL")
        logger.info("schema: knowledge_docs 补 content_hash 列（存量库迁移）")
