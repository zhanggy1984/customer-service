"""ToolGuardrail 实时护栏单测：每规则一正一反（fc-plan §P4 验收口径）。

护栏是纯确定性逻辑（无 IO），同步直接断言 GuardDecision 三态 + 理由。
"""
import pytest

from app.agent.function_calling import guardrail as gr


def test_side_effect_tool_reject():
    """[规则4正] 副作用工具 → reject(side_effect)：决策轮不该发起业务动作。"""
    g = gr.ToolGuardrail()
    d = g.check("create_return_order", {"order_id": "ORD-1", "items": ["SKU-1"]})
    assert d.verdict == "reject" and d.reason == "side_effect"


def test_readonly_tool_allow():
    """[规则4反] 只读工具正常决策 → allow。"""
    g = gr.ToolGuardrail()
    d = g.check("query_order", {"order_id": "ORD-1"})
    assert d.verdict == "allow"


def test_search_policy_too_short_reject():
    """[规则1正] search_policy query < 4 字 → reject(trivial_query)：检索无意义。"""
    g = gr.ToolGuardrail()
    d = g.check("search_policy", {"query": "退"})
    assert d.verdict == "reject" and d.reason == "trivial_query"


def test_search_policy_normal_allow():
    """[规则1反] 正常政策查询 → allow。"""
    g = gr.ToolGuardrail()
    d = g.check("search_policy", {"query": "退货政策是什么"})
    assert d.verdict == "allow"


def test_search_policy_greeting_reject():
    """[规则1正·纯问候] 纯问候 query → reject。"""
    g = gr.ToolGuardrail()
    d = g.check("search_policy", {"query": "你好"})
    assert d.verdict == "reject" and d.reason == "trivial_query"


def test_search_policy_greeting_embedded_not_rejected():
    """[规则1反·防误伤] 含问候词但非纯问候（"你好，退货政策是什么"）不被误伤。"""
    g = gr.ToolGuardrail()
    d = g.check("search_policy", {"query": "你好，退货政策是什么"})
    assert d.verdict == "allow"


def test_query_order_missing_id_override():
    """[规则3正] query_order 缺 order_id → override 为 list_user_orders 辅助定位。"""
    g = gr.ToolGuardrail()
    d = g.check("query_order", {})
    assert d.verdict == "override"
    assert d.tool_name == "list_user_orders"
    assert d.params == {"limit": 5}


def test_query_order_with_id_allow():
    """[规则3反] query_order 带 order_id → allow。"""
    g = gr.ToolGuardrail()
    d = g.check("query_order", {"order_id": "ORD-1"})
    assert d.verdict == "allow"


def test_dedupe_same_params_cached():
    """[规则5正] 同轮同工具同参数二次调用 → allow(dedupe) 命中首次结果缓存。"""
    g = gr.ToolGuardrail()
    r1 = {"order": {"order_id": "ORD-1", "status": "PAID"}}
    assert g.check("query_order", {"order_id": "ORD-1"}).verdict == "allow"
    g.record("query_order", {"order_id": "ORD-1"}, r1)
    d = g.check("query_order", {"order_id": "ORD-1"})
    assert d.verdict == "allow" and d.reason == "dedupe"
    assert d.cached_result == r1


def test_dedupe_different_params_not_cached():
    """[规则5反] 不同参数不命中缓存 → allow 且无缓存结果。"""
    g = gr.ToolGuardrail()
    g.record("query_order", {"order_id": "ORD-1"}, {"order": {"order_id": "ORD-1"}})
    d = g.check("query_order", {"order_id": "ORD-2"})
    assert d.verdict == "allow" and d.cached_result is None


def test_override_target_deduped():
    """[规则5·override 目标] query_order 无 id 二次决策 → override 目标命中缓存返回 dedupe。

    回归 hook 缺陷：dedupe 缓存键是原始 (query_order,{}) 时 override 目标永不命中，
    同参数二次决策会重复执行 list_user_orders。
    """
    g = gr.ToolGuardrail()
    r1 = {"orders": [{"order_id": "ORD-1", "status": "PAID"}]}
    g.record("list_user_orders", {"limit": 5}, r1)
    d = g.check("query_order", {})
    assert d.verdict == "allow" and d.reason == "dedupe"
    assert d.cached_result == r1


def test_call_count_limit_reject():
    """[规则6正] 累计调用达上限后 → reject(too_many_calls) 截断。"""
    g = gr.ToolGuardrail()
    for _ in range(gr.MAX_TOOL_CALLS):
        assert g.check("query_order", {"order_id": "ORD-1"}).verdict == "allow"
        g.record("query_order", {"order_id": "ORD-1"}, {"order": {}})
    d = g.check("query_order", {"order_id": "ORD-1"})
    assert d.verdict == "reject" and d.reason == "too_many_calls"


def test_call_count_under_limit_allow():
    """[规则6反] 未达上限 → 继续 allow。"""
    g = gr.ToolGuardrail()
    for _ in range(gr.MAX_TOOL_CALLS - 1):
        assert g.check("query_order", {"order_id": "ORD-1"}).verdict == "allow"
        g.record("query_order", {"order_id": "ORD-1"}, {"order": {}})
    assert g.check("query_order", {"order_id": "ORD-1"}).verdict == "allow"


def test_reject_not_counted():
    """reject 不累加调用计数：多次 reject 后正常工具仍可执行（截断只看真执行）。"""
    g = gr.ToolGuardrail()
    for _ in range(5):
        assert g.check("create_return_order", {}).verdict == "reject"
    assert g.check("query_order", {"order_id": "ORD-1"}).verdict == "allow"
