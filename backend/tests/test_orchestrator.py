"""编排器单元测试：多轮 items 合并 + 确认/取消时的 confirm action 语义（mock 分类与对接层）。

覆盖两个历史缺陷：
1. 取消（final）时仍发 confirm action → 前端确认按钮残留。根因修复：发 action 前排除 final。
2. 部分退货 items 仅首次注入，后轮"只退手机壳"被丢弃并误当原因，按全量确认。修复：推进前合并 items + 资格重算。
"""
import pytest

from app.agent.intent import IntentResult
from app.agent.orchestrator import run_agent
from app.services import order_service, return_service
from app.services.models import OrderInfo, OrderItem
from app.session.models import Session


def _order_multi():
    # 手机壳×1(29.9) + 钢化膜×2(19.9)，全部可退
    return OrderInfo(
        order_id="ORD-T", user_id=1, status="DELIVERED", total_amount=69.7, db_id=1,
        items=[
            OrderItem(id=1, item_id="SKU-1", name="手机壳", price=29.9, quantity=1, returnable=True),
            OrderItem(id=2, item_id="SKU-2", name="钢化膜", price=19.9, quantity=2, returnable=True),
        ],
    )


def _eligibility_all():
    return {"eligible": True, "reason": "", "refund_amount": 69.7, "items": _order_multi().items}


async def _fake_query(order_id, user_id):
    return _order_multi()


async def _fake_check(order, user_id):
    return _eligibility_all()


def _patch_services(monkeypatch):
    monkeypatch.setattr(order_service, "query_order", _fake_query)
    monkeypatch.setattr(return_service, "check_eligibility", _fake_check)


class _FakeClassify:
    """按轮次返回预设 IntentResult，记录每次输入。"""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    async def __call__(self, user_input, current_state_context=None, max_retries=2):
        self.calls.append(user_input)
        idx = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[idx]


class _Emit:
    def __init__(self):
        self.events = []

    async def __call__(self, evt):
        self.events.append(evt)

    def actions(self):
        return [e for e in self.events if e.get("type") == "action"]


def _session():
    return Session(session_id="s", user_id=1)


@pytest.mark.asyncio
async def test_confirm_action_emitted_on_confirm_node(monkeypatch):
    """到达确认节点（awaiting=confirm 且非 final）时发一次 confirm action。"""
    classify = _FakeClassify([
        IntentResult(intent="RETURN_REQUEST", confidence=0.95, slots={"order_id": "ORD-T"}),
        IntentResult(intent="RETURN_REQUEST", confidence=0.95, slots={}),
    ])
    monkeypatch.setattr("app.agent.orchestrator.classify_intent", classify)
    _patch_services(monkeypatch)

    emit = _Emit()
    session = _session()
    reply1 = await run_agent(session, "我要退货 ORD-T", 1, emit)
    assert "原因" in reply1  # 停在 collect_reason，未发 action

    reply2 = await run_agent(session, "质量问题", 1, emit)
    assert "确认" in reply2
    actions = emit.actions()
    assert len(actions) == 1
    assert actions[0]["action"] == "confirm"


@pytest.mark.asyncio
async def test_cancel_at_confirm_emits_no_confirm_action(monkeypatch):
    """取消（final）时绝不发 confirm action，且流程状态被清空。"""
    classify = _FakeClassify([
        IntentResult(intent="RETURN_REQUEST", confidence=0.95, slots={"order_id": "ORD-T"}),
        IntentResult(intent="RETURN_REQUEST", confidence=0.95, slots={}),
        IntentResult(intent="RETURN_REQUEST", confidence=0.95, slots={}),
    ])
    monkeypatch.setattr("app.agent.orchestrator.classify_intent", classify)
    _patch_services(monkeypatch)

    emit = _Emit()
    session = _session()
    await run_agent(session, "我要退货 ORD-T", 1, emit)
    await run_agent(session, "质量问题", 1, emit)

    reply3 = await run_agent(session, "取消", 1, emit)
    assert "取消" in reply3
    assert session.agent_state is None
    assert session.intent is None
    # 只允许确认节点发过的那一次，取消轮不得追加
    assert len(emit.actions()) == 1


@pytest.mark.asyncio
async def test_multi_round_items_merged_and_eligibility_recomputed(monkeypatch):
    """后轮补充指定退货商品：items 合并进 state，且按新子集重算资格（不全量确认）。"""
    classify = _FakeClassify([
        IntentResult(intent="RETURN_REQUEST", confidence=0.95, slots={"order_id": "ORD-T"}),
        IntentResult(intent="RETURN_REQUEST", confidence=0.95, slots={"items": ["手机壳"]}),
        IntentResult(intent="RETURN_REQUEST", confidence=0.95, slots={}),
    ])
    monkeypatch.setattr("app.agent.orchestrator.classify_intent", classify)
    _patch_services(monkeypatch)

    emit = _Emit()
    session = _session()
    await run_agent(session, "我要退货 ORD-T", 1, emit)
    assert session.agent_state["return_items"] == []

    await run_agent(session, "只退手机壳", 1, emit)
    # items 已合并，且资格按手机壳子集重算（回到 collect_reason 追原因）
    assert session.agent_state["return_items"] == ["手机壳"]
    assert [i["item_id"] for i in session.agent_state["eligibility"]["items"]] == ["SKU-1"]
    assert session.agent_state["eligibility"]["refund_amount"] == 29.9

    reply3 = await run_agent(session, "质量问题", 1, emit)
    assert "手机壳×1" in reply3
    assert "钢化膜" not in reply3
    assert "29.9" in reply3
    assert len(emit.actions()) == 1  # 确认 action 只在最终的确认节点发
