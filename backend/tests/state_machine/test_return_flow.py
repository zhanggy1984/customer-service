"""退货状态机单元测试（mock 对接层）。"""
import pytest

from app.agent.state_machine.return_flow import ReturnFlow
from app.services import order_service, return_service
from app.services.models import OrderInfo, OrderItem, ReturnResult


def _order():
    return OrderInfo(
        order_id="ORD-T", user_id=1, status="DELIVERED", total_amount=100.0, db_id=1,
        items=[OrderItem(id=1, item_id="SKU-1", name="商品", price=50.0, quantity=1, returnable=True)],
    )


@pytest.mark.asyncio
async def test_full_return_flow(monkeypatch):
    async def fake_query(order_id, user_id):
        return _order()

    async def fake_check(order, user_id):
        return {"eligible": True, "reason": "", "refund_amount": 50.0, "items": _order().items}

    async def fake_create(order, user_id, items, reason, session_id):
        return ReturnResult(success=True, return_id="RC-1", status="APPROVED", refund_amount=50.0, message="ok")

    monkeypatch.setattr(order_service, "query_order", fake_query)
    monkeypatch.setattr(return_service, "check_eligibility", fake_check)
    monkeypatch.setattr(return_service, "create_return", fake_create)

    flow = ReturnFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "stage": "verify_order"}

    state = await flow.step(state, "查一下")
    assert state.get("awaiting") == "reason"  # 追回原因

    state = await flow.step(state, "质量问题")
    assert state.get("awaiting") == "confirm"  # 确认信息

    state = await flow.step(state, "确认")
    assert state.get("final") is True
    assert "RC-1" in state.get("message", "")


@pytest.mark.asyncio
async def test_cancel_flow(monkeypatch):
    async def fake_query(order_id, user_id):
        return _order()

    async def fake_check(order, user_id):
        return {"eligible": True, "reason": "", "refund_amount": 50.0, "items": _order().items}

    monkeypatch.setattr(order_service, "query_order", fake_query)
    monkeypatch.setattr(return_service, "check_eligibility", fake_check)

    flow = ReturnFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "stage": "verify_order"}
    state = await flow.step(state, "x")
    state = await flow.step(state, "取消")
    assert state.get("final") is True
    assert "取消" in state.get("message", "")


@pytest.mark.asyncio
async def test_ineligible_reason(monkeypatch):
    async def fake_query(order_id, user_id):
        return _order()

    async def fake_check(order, user_id):
        return {"eligible": False, "reason": "已超过 7 天退货期，不可退", "refund_amount": 0.0, "items": []}

    monkeypatch.setattr(order_service, "query_order", fake_query)
    monkeypatch.setattr(return_service, "check_eligibility", fake_check)

    flow = ReturnFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "stage": "verify_order"}
    state = await flow.step(state, "x")
    assert state.get("final") is True
    assert "7 天" in state.get("message", "")


@pytest.mark.asyncio
async def test_order_not_found(monkeypatch):
    async def fake_query(order_id, user_id):
        return None

    monkeypatch.setattr(order_service, "query_order", fake_query)
    flow = ReturnFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "stage": "verify_order"}
    state = await flow.step(state, "x")
    assert state.get("final") is True
    assert "不存在" in state.get("message", "")


@pytest.mark.asyncio
async def test_user_isolation(monkeypatch):
    async def fake_query(order_id, user_id):
        # user_id=2 查别人的订单 → None
        return None if user_id == 2 else _order()

    monkeypatch.setattr(order_service, "query_order", fake_query)
    flow = ReturnFlow()
    state = {"user_id": 2, "session_id": "s", "order_id": "ORD-T", "stage": "verify_order"}
    state = await flow.step(state, "x")
    assert state.get("final") is True
