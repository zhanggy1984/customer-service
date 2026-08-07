"""仅退款状态机单元测试（三级判定：PAID 可退 / SHIPPED 拒收 / DELIVERED 走退货）。"""
import pytest

from app.agent.state_machine.refund_flow import RefundFlow
from app.infrastructure.mysql import mysql_pool
from app.services import order_service
from app.services.local_impl import LocalRefundService
from app.services.models import OrderInfo, OrderItem


def _order(status: str):
    return OrderInfo(
        order_id="ORD-T", user_id=1, status=status, total_amount=100.0, db_id=1,
        items=[OrderItem(id=1, item_id="SKU-1", name="商品", price=100.0, quantity=1, returnable=True)],
    )


def _order_mixed(status: str = "PAID"):
    # 混合订单：数据线×2(29.95, 可退) + 定制手机支架×1(29.95, 不可退)，整单 89.85
    return OrderInfo(
        order_id="ORD-T", user_id=1, status=status, total_amount=89.85, db_id=1,
        items=[
            OrderItem(id=1, item_id="SKU-1", name="数据线", price=29.95, quantity=2, returnable=True),
            OrderItem(id=2, item_id="SKU-2", name="定制手机支架", price=29.95, quantity=1, returnable=False),
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expect_final", [
    ("PAID", False),      # 可退，继续到 collect_reason
    ("SHIPPED", True),    # 需先拒收，拒绝
    ("DELIVERED", True),  # 必须走退货，拒绝
])
async def test_refund_eligibility_levels(monkeypatch, status, expect_final):
    async def fake_query(order_id, user_id):
        return _order(status)

    monkeypatch.setattr(order_service, "query_order", fake_query)

    flow = RefundFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "stage": "verify_order"}
    state = await flow.step(state, "x")
    if expect_final:
        assert state.get("final") is True
    else:
        assert state.get("awaiting") == "reason"


@pytest.mark.asyncio
async def test_refund_full_flow_paid(monkeypatch):
    async def fake_query(order_id, user_id):
        return _order("PAID")

    monkeypatch.setattr(order_service, "query_order", fake_query)

    flow = RefundFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "stage": "verify_order"}
    state = await flow.step(state, "x")          # verify → check → collect_reason
    state = await flow.step(state, "不想要了")    # reason → confirm
    assert state.get("awaiting") == "confirm"
    # create_refund 未 mock，此处只验证走到 confirm（金额展示）
    assert "100" in state.get("message", "")


@pytest.mark.asyncio
async def test_refund_amount_excludes_non_returnable(monkeypatch):
    """仅退款金额过滤定制商品：PAID 混合订单只退可退子集（89.85 - 29.95 = 59.9）。"""
    async def fake_query(order_id, user_id):
        return _order_mixed("PAID")

    monkeypatch.setattr(order_service, "query_order", fake_query)

    flow = RefundFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "stage": "verify_order"}
    state = await flow.step(state, "x")           # verify → check(真实 LocalRefundService) → collect_reason
    assert state.get("awaiting") == "reason"
    assert state["eligibility"]["amount"] == 59.9

    state = await flow.step(state, "不想要了")     # reason → confirm
    assert state.get("awaiting") == "confirm"
    assert "59.9" in state.get("message", "")
    assert "89.85" not in state.get("message", "")


@pytest.mark.asyncio
async def test_refund_rejected_all_non_returnable(monkeypatch):
    """仅退款全定制订单：所有商品 returnable=false → 拒绝退款。"""
    order = _order_mixed("PAID")
    order.items = [it for it in order.items if not it.returnable]  # 只剩定制支架

    async def fake_query(order_id, user_id):
        return order

    monkeypatch.setattr(order_service, "query_order", fake_query)

    flow = RefundFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "stage": "verify_order"}
    state = await flow.step(state, "x")
    assert state.get("final") is True
    assert "定制" in state.get("message", "")


@pytest.mark.asyncio
async def test_create_refund_amount_filters_non_returnable(monkeypatch):
    """create_refund 落库金额与资格判定同口径（过滤定制商品）。"""
    captured = {}

    async def fake_execute(sql, params):
        captured["sql"] = sql
        captured["params"] = params

    monkeypatch.setattr(mysql_pool, "execute", fake_execute)

    svc = LocalRefundService()
    result = await svc.create_refund(_order_mixed(), 1, "不想要了", "s")
    assert result.amount == 59.9
    # INSERT 参数顺序：refund_id, order_id, user_id, reason, amount, session_id
    assert captured["params"][4] == 59.9
