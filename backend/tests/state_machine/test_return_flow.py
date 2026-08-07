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


def _order_multi():
    # 多商品订单：手机壳×1(29.9) + 钢化膜×2(19.9) 全部可退
    return OrderInfo(
        order_id="ORD-T", user_id=1, status="DELIVERED", total_amount=69.7, db_id=1,
        items=[
            OrderItem(id=1, item_id="SKU-1", name="手机壳", price=29.9, quantity=1, returnable=True),
            OrderItem(id=2, item_id="SKU-2", name="钢化膜", price=19.9, quantity=2, returnable=True),
        ],
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


@pytest.mark.asyncio
async def test_partial_return_selected_items(monkeypatch):
    """部分退货：用户指定只退手机壳 → 只退该商品子集，金额按子集算，确认页不出现钢化膜。"""
    created = {}

    async def fake_query(order_id, user_id):
        return _order_multi()

    async def fake_check(order, user_id):
        return {"eligible": True, "reason": "", "refund_amount": 69.7, "items": _order_multi().items}

    async def fake_create(order, user_id, items, reason, session_id):
        created["items"] = items
        return ReturnResult(success=True, return_id="RC-1", status="APPROVED", refund_amount=29.9, message="ok")

    monkeypatch.setattr(order_service, "query_order", fake_query)
    monkeypatch.setattr(return_service, "check_eligibility", fake_check)
    monkeypatch.setattr(return_service, "create_return", fake_create)

    flow = ReturnFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "return_items": ["手机壳"], "stage": "verify_order"}

    state = await flow.step(state, "查一下")
    # 子集筛出手机壳，金额 29.9，进入 collect_reason
    assert state.get("awaiting") == "reason"
    assert state["eligibility"]["refund_amount"] == 29.9
    assert [i["item_id"] for i in state["eligibility"]["items"]] == ["SKU-1"]

    state = await flow.step(state, "质量问题")
    assert state.get("awaiting") == "confirm"
    assert "手机壳×1" in state.get("message", "")
    assert "钢化膜" not in state.get("message", "")

    state = await flow.step(state, "确认")
    assert state.get("final") is True
    assert created["items"] == ["SKU-1"]


@pytest.mark.asyncio
async def test_partial_return_selected_unavailable(monkeypatch):
    """部分退货：指定商品均不可退 → 明确拒绝，不回退全部商品。"""
    async def fake_query(order_id, user_id):
        return _order_multi()

    async def fake_check(order, user_id):
        return {"eligible": True, "reason": "", "refund_amount": 69.7, "items": _order_multi().items}

    monkeypatch.setattr(order_service, "query_order", fake_query)
    monkeypatch.setattr(return_service, "check_eligibility", fake_check)

    flow = ReturnFlow()
    state = {"user_id": 1, "session_id": "s", "order_id": "ORD-T", "return_items": ["不存在商品"], "stage": "verify_order"}
    state = await flow.step(state, "查一下")
    assert state.get("final") is True
    assert "不支持退货" in state.get("message", "")
    assert "手机壳" in state.get("message", "")  # 提示中给出可退清单
