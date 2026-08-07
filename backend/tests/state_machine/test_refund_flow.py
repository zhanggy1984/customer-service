"""仅退款状态机单元测试（三级判定：PAID 可退 / SHIPPED 拒收 / DELIVERED 走退货）。"""
import pytest

from app.agent.state_machine.refund_flow import RefundFlow
from app.services import order_service
from app.services.models import OrderInfo, OrderItem


def _order(status: str):
    return OrderInfo(
        order_id="ORD-T", user_id=1, status=status, total_amount=100.0, db_id=1,
        items=[OrderItem(id=1, item_id="SKU-1", name="商品", price=100.0, quantity=1, returnable=True)],
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
