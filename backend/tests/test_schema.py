"""init_schema 切分与执行测试（P3.3：共享 mysql 不跑 init.sql，应用启动自建表+种子）

守护修复：init.sql 每个语句前带 `-- ---------- xxx ----------` 注释行，切分不得按"段以
-- 开头"丢弃含 SQL 的段——否则 fresh infra 上建表+种子全被跳过（应用无表可用）。
"""
import asyncio

from app.infrastructure import schema


class _FakePool:
    """记录被执行的 SQL，不连真实 MySQL"""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.one: dict | None = {"c": 1}  # 默认 content_hash 列已存在 → 不触发 ALTER 迁移

    async def execute(self, sql: str) -> None:
        self.executed.append(sql)

    async def fetchone(self, sql: str, params: tuple | None = None) -> dict | None:
        return self.one


def test_init_schema_executes_all_statements(monkeypatch):
    """init.sql 全部 14 条语句（SET/USE + 9 建表 + 3 种子）均被切分并执行"""
    fake = _FakePool()
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema.init_schema())

    assert len(fake.executed) == 14, f"应为 14 条语句，实际 {len(fake.executed)}"
    joined = "\n".join(fake.executed)
    # 建表（抽查关键表）
    assert "CREATE TABLE IF NOT EXISTS users" in joined
    assert "CREATE TABLE IF NOT EXISTS orders" in joined
    assert "CREATE TABLE IF NOT EXISTS tool_call_log" in joined
    # 种子（幂等插入：用户/订单/订单项）
    assert "INSERT IGNORE INTO users" in joined
    assert "INSERT IGNORE INTO orders" in joined
    assert "INSERT INTO order_items" in joined
    # 库上下文
    assert "USE customer_service" in joined


def test_init_schema_skips_missing_file(monkeypatch):
    """init.sql 不存在时不抛错，仅告警跳过"""
    fake = _FakePool()
    monkeypatch.setattr(schema, "mysql_pool", fake)
    monkeypatch.setattr(
        schema, "INIT_SQL_PATH", schema.INIT_SQL_PATH.parent / "no-such-init.sql"
    )
    asyncio.run(schema.init_schema())
    assert fake.executed == []


def test_ensure_hash_column_migrates_when_missing(monkeypatch):
    """存量库缺 content_hash 列 → ALTER 补列（增量跳检判据）"""
    fake = _FakePool()
    fake.one = {"c": 0}
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_knowledge_hash_column())
    assert any(
        "ALTER TABLE knowledge_docs ADD COLUMN content_hash" in s
        for s in fake.executed
    )


def test_ensure_hash_column_skips_when_present(monkeypatch):
    """content_hash 列已存在 → 不执行 ALTER（幂等，避免重复迁移报错）"""
    fake = _FakePool()
    fake.one = {"c": 1}
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_knowledge_hash_column())
    assert not any("ALTER TABLE knowledge_docs" in s for s in fake.executed)


def test_ensure_refund_order_unique_migrates_when_missing(monkeypatch):
    """存量库缺 uk_refund_order_user 且无重复数据 → ALTER ADD UNIQUE（写路径幂等）"""
    fake = _FakePool()

    async def fake_fetchone(sql: str) -> dict | None:
        # 第一次查 STATISTICS 索引=0（缺失），第二次查重复行=0（无重复）
        return {"c": 0}

    fake.fetchone = fake_fetchone
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_refund_order_unique())
    assert any(
        "ALTER TABLE refund_orders ADD UNIQUE KEY uk_refund_order_user" in s
        for s in fake.executed
    )


def test_ensure_refund_order_unique_skips_when_present(monkeypatch):
    """uk_refund_order_user 已存在 → 不执行 ALTER"""
    fake = _FakePool()
    fake.one = {"c": 1}
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_refund_order_unique())
    assert not any("ALTER TABLE refund_orders" in s for s in fake.executed)


def test_ensure_refund_order_unique_skips_on_duplicates(monkeypatch):
    """存量有重复 (order_id,user_id) → 告警跳过不 ALTER（避免 ADD UNIQUE 失败阻断启动）"""
    fake = _FakePool()

    async def fake_fetchone(sql: str) -> dict | None:
        return {"c": 1} if "STATISTICS" in sql else {"c": 2}  # 索引缺失 + 重复 2 组

    fake.fetchone = fake_fetchone
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_refund_order_unique())
    assert not any("ALTER TABLE refund_orders" in s for s in fake.executed)


def test_ensure_ticket_idempotency_key_migrates_when_missing(monkeypatch):
    """存量库缺 idempotency_key 列 → ALTER 补列 + 唯一约束（complaint 写路径幂等）"""
    fake = _FakePool()
    fake.one = {"c": 0}
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_ticket_idempotency_key())
    assert any(
        "ALTER TABLE complaint_tickets ADD COLUMN idempotency_key" in s
        for s in fake.executed
    )


def test_ensure_ticket_idempotency_key_skips_when_present(monkeypatch):
    """idempotency_key 列已存在 → 不执行 ALTER"""
    fake = _FakePool()
    fake.one = {"c": 1}
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_ticket_idempotency_key())
    assert not any("ALTER TABLE complaint_tickets" in s for s in fake.executed)


def test_ensure_created_at_index_migrates_when_missing(monkeypatch):
    """存量库缺 idx_created_at → 两表各 ALTER 补索引（TTL 清理按时间删除，缺索引会全表扫）"""
    fake = _FakePool()
    fake.one = {"c": 0}
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_created_at_index())
    assert any(
        "ALTER TABLE conversation_history ADD KEY idx_created_at" in s
        for s in fake.executed
    )
    assert any(
        "ALTER TABLE tool_call_log ADD KEY idx_created_at" in s
        for s in fake.executed
    )


def test_ensure_created_at_index_skips_when_present(monkeypatch):
    """idx_created_at 已存在 → 不执行 ALTER（幂等）"""
    fake = _FakePool()
    fake.one = {"c": 1}
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_created_at_index())
    assert not any("ADD KEY idx_created_at" in s for s in fake.executed)
