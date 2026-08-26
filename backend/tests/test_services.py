"""服务层重试/幂等/熔断测试。

- 读路径重试：query_order/list_user_orders 连接类错误收敛为 ServiceUnavailableException。
- 写路径幂等：create_refund/create_complaint 重复提交（uk 冲突 rowcount=0）返回已有单号。
- DB 熔断：连续失败达阈值 open，后续调用快速失败不重试（防重试风暴）。
"""
import pytest
from asyncmy.errors import IntegrityError, OperationalError

from app.services import local_impl
from app.services.exceptions import ServiceUnavailableException
from app.services.local_impl import (
    LocalComplaintService,
    LocalOrderService,
    LocalRefundService,
    LocalReturnService,
)
from app.services.models import OrderInfo, OrderItem


def _order():
    """可退订单：单商品手机壳，供 create_refund/create_complaint 幂等测试。"""
    return OrderInfo(
        order_id="ORD-T", user_id=1, status="DELIVERED", total_amount=29.9, db_id=1,
        items=[OrderItem(id=1, item_id="SKU-1", name="手机壳", price=29.9, quantity=1, returnable=True)],
    )


@pytest.mark.asyncio
async def test_query_order_operational_error_retries_then_unavailable(monkeypatch):
    """query_order 连接类读错误：重试 3 次耗尽后抛 ServiceUnavailableException（非裸 OperationalError）。"""
    calls = {"n": 0}

    async def fake_fetchone(sql, params):
        calls["n"] += 1
        raise OperationalError("connection reset")

    async def fake_fetchall(sql, params):
        raise AssertionError("fetchone 失败不应走到 _load_items")

    monkeypatch.setattr(local_impl.mysql_pool, "fetchone", fake_fetchone)
    monkeypatch.setattr(local_impl.mysql_pool, "fetchall", fake_fetchall)

    svc = LocalOrderService()
    with pytest.raises(ServiceUnavailableException):
        await svc.query_order("ORD-1", 1)
    assert calls["n"] == 4  # 原始 1 次 + 指数退避重试 3 次


@pytest.mark.asyncio
async def test_list_user_orders_operational_error_retries_then_unavailable(monkeypatch):
    """list_user_orders 连接类读错误同样收敛为 ServiceUnavailableException。"""
    calls = {"n": 0}

    async def fake_fetchall(sql, params):
        calls["n"] += 1
        raise OperationalError("connection reset")

    async def fake_fetchone(sql, params):
        raise AssertionError("不应走到 fetchone")

    monkeypatch.setattr(local_impl.mysql_pool, "fetchall", fake_fetchall)
    monkeypatch.setattr(local_impl.mysql_pool, "fetchone", fake_fetchone)

    svc = LocalOrderService()
    with pytest.raises(ServiceUnavailableException):
        await svc.list_user_orders(1)
    assert calls["n"] == 4


@pytest.mark.asyncio
async def test_create_refund_uses_insert(monkeypatch):
    """create_refund 正常插入：INSERT（非 IGNORE，避免吞非唯一错误），插入成功不查已有。"""
    captured = {}

    async def fake_execute(sql, params):
        captured["sql"] = sql
        return 1  # rowcount=1 插入成功

    async def fake_fetchone(sql, params):
        raise AssertionError("插入成功不应查已有")

    monkeypatch.setattr(local_impl.mysql_pool, "execute", fake_execute)
    monkeypatch.setattr(local_impl.mysql_pool, "fetchone", fake_fetchone)

    svc = LocalRefundService()
    result = await svc.create_refund(_order(), 1, "不想要了", "s")
    assert result.success is True
    assert "INSERT INTO" in captured["sql"]
    assert "INSERT IGNORE" not in captured["sql"]


@pytest.mark.asyncio
async def test_create_refund_duplicate_returns_existing(monkeypatch):
    """create_refund 重复提交（uk_refund_order_user 唯一键冲突）→ 幂等返回已有单号，不重复创建。"""
    async def fake_execute(sql, params):
        raise IntegrityError(1062, "Duplicate entry '1-1' for key 'uk_refund_order_user'")

    async def fake_fetchone(sql, params):
        return {"refund_id": "RF-123"}

    monkeypatch.setattr(local_impl.mysql_pool, "execute", fake_execute)
    monkeypatch.setattr(local_impl.mysql_pool, "fetchone", fake_fetchone)

    svc = LocalRefundService()
    result = await svc.create_refund(_order(), 1, "不想要了", "s")
    assert result.success is True
    assert result.refund_id == "RF-123"  # 返回已有单号而非新建
    assert "已提交过" in result.message


@pytest.mark.asyncio
async def test_create_complaint_uses_insert_with_idempotency_key(monkeypatch):
    """create_complaint 正常插入：INSERT（非 IGNORE）+ idempotency_key 列（幂等基础）。"""
    captured = {}

    async def fake_execute(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return 1

    async def fake_fetchone(sql, params):
        raise AssertionError("插入成功不应查已有")

    monkeypatch.setattr(local_impl.mysql_pool, "execute", fake_execute)
    monkeypatch.setattr(local_impl.mysql_pool, "fetchone", fake_fetchone)

    svc = LocalComplaintService()
    result = await svc.create_complaint(
        user_id=1, order_id="ORD-1", complaint_type="物流问题",
        description="快递丢失", severity="MEDIUM", session_id="s",
    )
    assert result.success is True
    assert "INSERT INTO" in captured["sql"]
    assert "INSERT IGNORE" not in captured["sql"]
    assert "idempotency_key" in captured["sql"]
    # params 末尾是幂等键：确定性派生（同参数重复调用 key 相同），定长 ≤ VARCHAR(64)
    key = local_impl._ticket_idempotency_key(1, "ORD-1", "物流问题", "快递丢失")
    assert captured["params"][-1] == key
    assert len(key) <= 64


@pytest.mark.asyncio
async def test_create_complaint_duplicate_returns_existing(monkeypatch):
    """create_complaint 重复提交（uk_ticket_idempotency 唯一键冲突）→ 幂等返回已有工单号。"""
    async def fake_execute(sql, params):
        raise IntegrityError(1062, "Duplicate entry 'xxx' for key 'uk_ticket_idempotency'")

    async def fake_fetchone(sql, params):
        return {"ticket_id": "CT-999"}

    monkeypatch.setattr(local_impl.mysql_pool, "execute", fake_execute)
    monkeypatch.setattr(local_impl.mysql_pool, "fetchone", fake_fetchone)

    svc = LocalComplaintService()
    result = await svc.create_complaint(
        user_id=1, order_id="ORD-1", complaint_type="物流问题",
        description="快递丢失", severity="MEDIUM", session_id="s",
    )
    assert result.success is True
    assert result.ticket_id == "CT-999"
    assert "已提交过" in result.message


@pytest.mark.asyncio
async def test_db_breaker_opens_after_threshold(monkeypatch):
    """DB 熔断：连续失败达阈值 → open，后续调用快速失败不再重试（防重试风暴）。"""
    from app.services import retry as retry_mod

    monkeypatch.setattr(retry_mod, "BACKOFF_DELAYS", (0.0, 0.0, 0.0))  # 免退避等待
    calls = {"n": 0}

    async def fake_fetchone(sql, params):
        calls["n"] += 1
        raise OperationalError("connection refused")

    async def fake_fetchall(sql, params):
        raise AssertionError("不应走到 fetchall")

    monkeypatch.setattr(local_impl.mysql_pool, "fetchone", fake_fetchone)
    monkeypatch.setattr(local_impl.mysql_pool, "fetchall", fake_fetchall)

    svc = LocalOrderService()
    retry_mod._breaker["failures"] = retry_mod.DB_BREAKER_FAIL_THRESHOLD - 1  # 预置到阈值-1

    with pytest.raises(ServiceUnavailableException):
        await svc.query_order("ORD-1", 1)  # 第 1 次失败即达阈值 → 熔断 open
    assert retry_mod._breaker["failures"] >= retry_mod.DB_BREAKER_FAIL_THRESHOLD

    calls_before = calls["n"]
    with pytest.raises(ServiceUnavailableException):
        await svc.query_order("ORD-1", 1)  # 熔断 open → 快速失败
    assert calls["n"] == calls_before  # 零 DB 尝试，不再 4 连打


@pytest.mark.asyncio
async def test_create_return_duplicate_returns_existing(monkeypatch):
    """create_return 重复退货（uk_return_order_user 唯一键冲突）→ 幂等返回已有退货单号（不假拒绝）。"""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_transaction():
        async def run(sql, params=None):
            raise IntegrityError(1062, "Duplicate entry '1-1' for key 'uk_return_order_user'")
        yield run

    async def fake_fetchone(sql, params):
        return {"return_id": "RC-999"}

    monkeypatch.setattr(local_impl.mysql_pool, "transaction", fake_transaction)
    monkeypatch.setattr(local_impl.mysql_pool, "fetchone", fake_fetchone)

    svc = LocalReturnService()
    result = await svc.create_return(_order(), 1, ["SKU-1"], "不想要了", "s")
    assert result.success is True
    assert result.return_id == "RC-999"  # 返回已有单号而非假拒绝
    assert "已提交过" in result.message


@pytest.mark.asyncio
async def test_create_refund_id_collision_raises_unavailable(monkeypatch):
    """唯一键冲突但幂等查询查不到（refund_id 时间戳碰撞等非幂等冲突）→ ServiceUnavailableException，不回落假成功。"""
    async def fake_execute(sql, params):
        raise IntegrityError(1062, "Duplicate entry 'xxx' for key 'PRIMARY'")

    async def fake_fetchone(sql, params):
        return None  # (order_id, user_id) 查不到 → 非幂等冲突

    monkeypatch.setattr(local_impl.mysql_pool, "execute", fake_execute)
    monkeypatch.setattr(local_impl.mysql_pool, "fetchone", fake_fetchone)

    svc = LocalRefundService()
    with pytest.raises(ServiceUnavailableException):
        await svc.create_refund(_order(), 1, "不想要了", "s")
