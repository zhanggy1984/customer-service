"""tool_call_log 落库单测（P5）。

核心逻辑：_result_summary 摘要函数（纯函数）+ write_tool_call 的 INSERT 组装与失败静默。
不连真实 MySQL：mock mysql_pool.execute。
"""
import pytest

from app.agent.function_calling import tool_call_log as tcl
from app.infrastructure.mysql import mysql_pool


def test_result_summary_results_list():
    """results 列表 → 命中数摘要。"""
    assert tcl._result_summary({"results": [{}, {}, {}]}) == "命中 3 条"


def test_result_summary_orders():
    """orders 列表 → 订单数摘要。"""
    assert tcl._result_summary({"orders": [{}]}) == "1 条订单"


def test_result_summary_order_dict():
    """单订单 dict → 订单号 + 状态摘要。"""
    assert tcl._result_summary({"order": {"order_id": "ORD-1", "status": "PAID"}}) == "订单 ORD-1 状态 PAID"


def test_result_summary_none_or_empty():
    """空结果 → 空串（=工具空结果，admin 聚合据此算空结果率）。"""
    assert tcl._result_summary(None) == ""
    assert tcl._result_summary({}) == ""


def test_result_summary_fallback_json():
    """其他结构 → JSON 截断兜底。"""
    s = tcl._result_summary({"foo": "bar"})
    assert s.startswith("{") and "bar" in s


@pytest.mark.asyncio
async def test_write_tool_call_insert(monkeypatch):
    """正常落库：组装 INSERT 与参数，query_text 截断前 500 字。"""
    calls = {}

    async def fake_execute(sql, params):
        calls["sql"] = sql
        calls["params"] = params

    monkeypatch.setattr(mysql_pool, "execute", fake_execute)
    long_query = "查" * 600
    await tcl.write_tool_call(
        session_id="sess-1", user_id=1, round_no=1, tool_name="search_policy",
        args={"query": "退货政策"}, result={"results": [{"text": "x"}]}, latency_ms=12,
        verdict="allow", reason="", query_text=long_query,
    )
    assert "INSERT INTO tool_call_log" in calls["sql"]
    sid, uid, rnd, tool, args, summary, latency, verdict, reason, query = calls["params"]
    assert (sid, uid, rnd, tool) == ("sess-1", 1, 1, "search_policy")
    assert '"query": "退货政策"' in args  # args_json 为 JSON 字符串
    assert summary == "命中 1 条"
    assert (latency, verdict, reason) == (12, "allow", "")
    assert len(query) == 500  # 截断


@pytest.mark.asyncio
async def test_write_tool_call_failure_silent(monkeypatch):
    """落库失败静默：异常仅吞掉不抛出，观测层不阻断决策主流程。"""
    async def fake_execute(sql, params):
        raise RuntimeError("db down")

    monkeypatch.setattr(mysql_pool, "execute", fake_execute)
    # 不抛异常即通过
    await tcl.write_tool_call(
        session_id="s", user_id=1, round_no=1, tool_name="query_order",
        args={}, result=None, latency_ms=0, verdict="reject", reason="side_effect", query_text="q",
    )
