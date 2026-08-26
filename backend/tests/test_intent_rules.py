"""意图规则层单元测试（纯正则，不涉 LLM）。

覆盖 match_intent_rules 的顺序求值语义：各类命中、slots/items 提取、疑问句与
政策咨询回退 LLM、动作规则优先于订单状态（防"物流"语义词抢走投诉）。
"""
import pytest

from app.agent.intent_rules import match_intent_rules


@pytest.mark.parametrize("text", ["你好", "您好", "hi", "Hello", "在吗", "哈喽", "晚上好", "你好！"])
def test_greeting_variants_chitchat(text):
    r = match_intent_rules(text)
    assert r is not None and r.intent == "CHITCHAT"


def test_mixed_greeting_not_chitchat():
    # 问候锚定整句：混合句不命中，落到动作规则
    r = match_intent_rules("你好，我要退货")
    assert r is not None and r.intent == "RETURN_REQUEST"


def test_order_status_with_order_id():
    r = match_intent_rules("查一下订单 ORD-20240805-002 到哪了")
    assert r is not None and r.intent == "ORDER_STATUS"
    assert r.slots["order_id"] == "ORD-20240805-002"
    assert r.missing_slots == []


def test_order_status_bare_query():
    r = match_intent_rules("查订单")
    assert r is not None and r.intent == "ORDER_STATUS"
    assert r.missing_slots == ["order_id"]


def test_order_status_question_form():
    # 疑问式状态查询：先于疑问门命中
    r = match_intent_rules("订单什么时候到")
    assert r is not None and r.intent == "ORDER_STATUS"


def test_order_id_with_status_semantic():
    r = match_intent_rules("ORD-20240805-002 到哪了")
    assert r is not None and r.intent == "ORDER_STATUS"
    assert r.slots["order_id"] == "ORD-20240805-002"


def test_order_id_alone_not_taken():
    # 裸订单号无查询语义词：保守不接管（可能是退货/退款/投诉引用）
    assert match_intent_rules("ORD-20240801-001") is None


def test_return_request_with_order_id():
    r = match_intent_rules("我要退货 ORD-20240801-001")
    assert r is not None and r.intent == "RETURN_REQUEST"
    assert r.slots["order_id"] == "ORD-20240801-001"
    assert r.missing_slots == []


def test_return_request_missing_order_id():
    r = match_intent_rules("我要退货")
    assert r is not None and r.intent == "RETURN_REQUEST"
    assert r.missing_slots == ["order_id"]


def test_partial_return_items():
    r = match_intent_rules("我要退货，只退手机壳")
    assert r is not None and r.intent == "RETURN_REQUEST"
    assert r.slots["items"] == [{"name": "手机壳"}]


def test_refund_request():
    r = match_intent_rules("申请退款")
    assert r is not None and r.intent == "REFUND_REQUEST"
    assert r.missing_slots == ["order_id"]


def test_refund_request_with_order_id():
    r = match_intent_rules("我要退款 ORD-20240806-003")
    assert r is not None and r.intent == "REFUND_REQUEST"
    assert r.slots["order_id"] == "ORD-20240806-003"


def test_complaint():
    r = match_intent_rules("我要投诉")
    assert r is not None and r.intent == "COMPLAINT"
    assert r.missing_slots == []


def test_complaint_not_taken_by_order_status():
    # 顺序守卫：动作规则先于订单状态，防"物流/到哪"语义词抢走投诉
    r = match_intent_rules("ORD-20240805-002 物流太慢，我要投诉")
    assert r is not None and r.intent == "COMPLAINT"


def test_policy_adjacent_action_not_misjudged():
    # 前缀紧邻守卫：动作前缀与"退货"间有内容（查一下）→ 政策咨询，不接管
    assert match_intent_rules("帮我查一下退货政策") is None


@pytest.mark.parametrize(
    "text",
    [
        "能退货吗",
        "可以退款吗",
        "退款怎么操作",
        "退货政策是什么",
        "能只退款不退货吗",
        "订单可以退款吗",
    ],
)
def test_question_forms_fall_back_to_llm(text):
    # 疑问句式=政策/资格咨询语义，一律回退 LLM
    assert match_intent_rules(text) is None


def test_unmatched_chitchat():
    assert match_intent_rules("随便聊聊天气怎么样") is None
