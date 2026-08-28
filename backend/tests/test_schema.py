"""init_schema 切分与执行测试（P3.3：共享 mysql 不跑 init.sql，应用启动自建表+种子）

守护修复：init.sql 每个语句前带 `-- ---------- xxx ----------` 注释行，切分不得按"段以
-- 开头"丢弃含 SQL 的段——否则 fresh infra 上建表+种子全被跳过（应用无表可用）。
"""
import asyncio

import pytest

from app.infrastructure import schema


class _FakePool:
    """记录被执行的 SQL，不连真实 MySQL"""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.one: dict | None = {"c": 1}  # 默认 content_hash 列已存在 → 不触发 ALTER 迁移
        # admin 查询（SELECT ... FROM users WHERE username）返回：None=不存在→走创建分支
        self.admin_row: dict | None = None

    async def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed.append(sql)

    async def fetchone(self, sql: str, params: tuple | None = None) -> dict | None:
        if "FROM users WHERE username" in sql:
            return self.admin_row
        return self.one


def test_init_schema_executes_all_statements(monkeypatch):
    """init.sql 全部 14 条语句（SET/USE + 9 建表 + 3 种子）+ admin 创建（共 15 条）均被切分并执行"""
    from app.config import settings

    fake = _FakePool()
    monkeypatch.setattr(schema, "mysql_pool", fake)
    # init_schema 末尾必走 _ensure_admin_password（fail-fast：空密码拒绝启动），其读
    # settings.admin_default_password。CI/干净环境未设 ADMIN_DEFAULT_PASSWORD 即空 →
    # RuntimeError；测试必须自包含 mock 非空（与其余 admin 测试一致），不隐式依赖运行环境 env。
    monkeypatch.setattr(settings, "admin_default_password", "str0ng-admin-pw")
    asyncio.run(schema.init_schema())

    assert len(fake.executed) == 15, f"应为 15 条语句（14 init.sql + 1 admin 创建），实际 {len(fake.executed)}"
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


# ---------- _ensure_admin_password（admin 弱口令安全兜底）----------

def test_ensure_admin_password_creates_when_missing(monkeypatch):
    """users 表无 admin → 按 env 密码创建 admin（不再依赖 init.sql 弱口令种子）"""
    from app.config import settings

    monkeypatch.setattr(settings, "admin_default_username", "admin")
    monkeypatch.setattr(settings, "admin_default_password", "str0ng-admin-pw")
    fake = _FakePool()
    fake.admin_row = None  # admin 不存在 → 走创建分支
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_admin_password())
    inserts = [s for s in fake.executed if "INSERT INTO users" in s]
    assert len(inserts) == 1
    assert "INSERT INTO users (username" in inserts[0]  # 按 env 用户名参数化插入


def test_ensure_admin_password_overrides_weak_hash(monkeypatch):
    """存量 admin 是弱口令 hash（历史 init.sql 种子/未知弱口令）→ 覆盖为 env 密码"""
    from app.config import settings

    monkeypatch.setattr(settings, "admin_default_username", "admin")
    monkeypatch.setattr(settings, "admin_default_password", "str0ng-admin-pw")
    fake = _FakePool()
    fake.admin_row = {"id": 7, "password_hash": "$2b$12$weakhashhistoricalseed"}
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_admin_password())
    assert any("UPDATE users SET password_hash" in s for s in fake.executed)


def test_ensure_admin_password_overrides_strong_hash(monkeypatch):
    """存量 admin 已是强 hash 也覆盖——env 是唯一事实来源（本系统无改密入口，无条件同步）"""
    from app.config import settings

    monkeypatch.setattr(settings, "admin_default_username", "admin")
    monkeypatch.setattr(settings, "admin_default_password", "str0ng-admin-pw")
    fake = _FakePool()
    fake.admin_row = {"id": 7, "password_hash": "$2b$12$somestronghashnotweak"}
    monkeypatch.setattr(schema, "mysql_pool", fake)
    asyncio.run(schema._ensure_admin_password())
    assert any("UPDATE users SET password_hash" in s for s in fake.executed)


def test_ensure_admin_password_empty_env_rejects(monkeypatch):
    """env 密码为空 → RuntimeError 拒绝启动（fail-fast 兜底，防误用）"""
    from app.config import settings

    monkeypatch.setattr(settings, "admin_default_username", "admin")
    monkeypatch.setattr(settings, "admin_default_password", "")
    fake = _FakePool()
    monkeypatch.setattr(schema, "mysql_pool", fake)
    with pytest.raises(RuntimeError, match="ADMIN_DEFAULT_PASSWORD"):
        asyncio.run(schema._ensure_admin_password())


# ---------- validate_security_config（启动 fail-fast 强校验）----------

def test_validate_security_config_rejects_weak_jwt(monkeypatch):
    """JWT 密钥为 change-me → 拒绝启动（伪造 token 风险，无逃生）"""
    from app.config import settings, validate_security_config

    monkeypatch.setattr(settings, "jwt_secret_key", "change-me")
    monkeypatch.setattr(settings, "admin_default_password", "str0ng-admin-pw")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        validate_security_config()


def test_validate_security_config_rejects_weak_admin(monkeypatch):
    """admin 密码为弱口令(admin123) 且未开逃生 → 拒绝启动"""
    from app.config import settings, validate_security_config

    monkeypatch.setattr(settings, "jwt_secret_key", "a" * 48)
    monkeypatch.setattr(settings, "admin_default_password", "admin123")
    monkeypatch.setattr(settings, "allow_weak_admin_password", False)
    with pytest.raises(RuntimeError, match="ADMIN_DEFAULT_PASSWORD"):
        validate_security_config()


def test_validate_security_config_accepts_strong(monkeypatch):
    """JWT + admin 密码均强 → 启动校验通过"""
    from app.config import settings, validate_security_config

    monkeypatch.setattr(settings, "jwt_secret_key", "a" * 48)
    monkeypatch.setattr(settings, "admin_default_password", "str0ng-admin-pw")
    validate_security_config()  # 不抛即通过


def test_validate_security_config_allow_weak_admin_escape(monkeypatch):
    """显式 ALLOW_WEAK_ADMIN_PASSWORD=true + 非生产(APP_ENV=dev) → 弱 admin 口令放行（演示逃生）"""
    from app.config import settings, validate_security_config

    monkeypatch.setattr(settings, "jwt_secret_key", "a" * 48)
    monkeypatch.setattr(settings, "admin_default_password", "admin123")
    monkeypatch.setattr(settings, "app_env", "dev")
    monkeypatch.setattr(settings, "allow_weak_admin_password", True)
    validate_security_config()  # 逃生开关放行


def test_validate_security_config_weak_admin_rejected_in_prod(monkeypatch):
    """生产(APP_ENV=prod)下逃生开关强制失效：弱 admin 口令仍拒绝启动"""
    from app.config import settings, validate_security_config

    monkeypatch.setattr(settings, "jwt_secret_key", "a" * 48)
    monkeypatch.setattr(settings, "admin_default_password", "admin123")
    monkeypatch.setattr(settings, "app_env", "prod")
    monkeypatch.setattr(settings, "allow_weak_admin_password", True)
    with pytest.raises(RuntimeError, match="ADMIN_DEFAULT_PASSWORD"):
        validate_security_config()


def test_settings_maps_app_env_from_env_var(monkeypatch):
    """APP_ENV 环境变量正确映射到 settings.app_env——防字段名↔env 名不一致（踩过的坑：
    字段曾叫 deploy_env 映射 DEPLOY_ENV，导致 APP_ENV=prod 不生效、逃生开关误放行）"""
    from app.config import Settings

    monkeypatch.setenv("APP_ENV", "prod")
    s = Settings()
    assert s.app_env == "prod"
