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
    await _ensure_refund_order_unique()
    await _ensure_ticket_idempotency_key()
    await _ensure_created_at_index()
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


async def _ensure_refund_order_unique() -> None:
    """存量库迁移：refund_orders 补 uk_refund_order_user 唯一约束（写路径幂等防重复退款）。

    CREATE TABLE IF NOT EXISTS 不会给已存在表加约束，故用 information_schema.STATISTICS
    检查索引缺失则 ALTER。ADD UNIQUE 前先查重复：存量若已有同 (order_id,user_id) 重复行，
    ALTER 会失败，此时仅告警跳过（重复需人工清理），不阻断启动。
    """
    row = await mysql_pool.fetchone(
        "SELECT COUNT(*) AS c FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'refund_orders' "
        "AND INDEX_NAME = 'uk_refund_order_user'"
    )
    if (row or {}).get("c", 0) > 0:
        return
    dup = await mysql_pool.fetchone(
        "SELECT COUNT(*) AS c FROM ("
        " SELECT order_id, user_id FROM refund_orders "
        " GROUP BY order_id, user_id HAVING COUNT(*) > 1"
        ") t"
    )
    if (dup or {}).get("c", 0) > 0:
        logger.warning("schema: refund_orders 存在重复 (order_id,user_id)，需人工清理后重启再迁移 uk_refund_order_user")
        return
    await mysql_pool.execute(
        "ALTER TABLE refund_orders ADD UNIQUE KEY uk_refund_order_user (order_id, user_id)"
    )
    logger.info("schema: refund_orders 补 uk_refund_order_user 唯一约束（存量库迁移）")


async def _ensure_ticket_idempotency_key() -> None:
    """存量库迁移：complaint_tickets 补 idempotency_key 列 + 唯一约束（写路径幂等防重复工单）。

    存量行该列全 NULL，UNIQUE 索引对 NULL 允许多行，ADD 安全不冲突。
    """
    row = await mysql_pool.fetchone(
        "SELECT COUNT(*) AS c FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'complaint_tickets' "
        "AND COLUMN_NAME = 'idempotency_key'"
    )
    if (row or {}).get("c", 0) == 0:
        await mysql_pool.execute(
            "ALTER TABLE complaint_tickets ADD COLUMN idempotency_key VARCHAR(64) NULL, "
            "ADD UNIQUE KEY uk_ticket_idempotency (idempotency_key)"
        )
        logger.info("schema: complaint_tickets 补 idempotency_key 列 + 唯一约束（存量库迁移）")


async def _ensure_created_at_index() -> None:
    """存量库迁移：conversation_history / tool_call_log 补 idx_created_at（TTL 清理按时间范围删除）。

    CREATE TABLE IF NOT EXISTS 不会给已存在表加索引，旧库两表缺 created_at 索引时
    清理 DELETE 会全表扫。用 information_schema.STATISTICS 检查缺则 ALTER，幂等。
    """
    for table in ("conversation_history", "tool_call_log"):
        row = await mysql_pool.fetchone(
            "SELECT COUNT(*) AS c FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s "
            "AND INDEX_NAME = 'idx_created_at'",
            (table,),
        )
        if (row or {}).get("c", 0) == 0:
            await mysql_pool.execute(f"ALTER TABLE {table} ADD KEY idx_created_at (created_at)")
            logger.info("schema: %s 补 idx_created_at 索引（存量库迁移）", table)
