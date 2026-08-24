"""agent_loop 决策循环单测（mock LLM 与工具执行，不真实调用）。

覆盖 P3 决策循环正反路径：
- 政策命中 → route=policy，search_policy 结果回灌 + tool_events 透出
- 订单查询 → route=order
- 无工具直接作答 → direct_reply 透出 + 按意图兜底路由
- 副作用工具决策 → 护栏拦截，route=business，工具不执行、不透出事件
- query_order 缺 order_id → override 为 list_user_orders
- FORCE_POLICY_SEARCH 闸门 → 强制补 search_policy
"""
import json

import pytest

from app.agent import agent_loop as al

POLICY_USAGE = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
                "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}


def _resp(content: str = "", tool_calls: list | None = None) -> dict:
    msg: dict = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}], "usage": dict(POLICY_USAGE)}


def _tool(name: str, args: dict, cid: str = "call_1") -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class _Session:
    session_id = "sess-1"


@pytest.mark.asyncio
async def test_policy_hit_loop(monkeypatch):
    """政策命中：首轮决策 search_policy 并执行，次轮不再调工具 → route=policy。"""
    calls = {"execute": []}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        return {"results": [{"text": "签收后 7 天内支持无理由退货。", "score": 0.9, "source": "x.md"}]}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="基于政策文档回答...")
        return _resp(tool_calls=[_tool("search_policy", {"query": "退货政策是什么？"})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    out = await al.run_decision_loop("退货政策是什么？", "POLICY_INQUIRY", _Session(), 1)
    assert out["route"] == "policy"
    assert "search_policy" in out["tool_results"]
    assert calls["execute"][0][0] == "search_policy"
    # 工具动作事件透出（事件名=工具名）
    assert out["tool_events"][0]["name"] == "search_policy"
    assert out["tool_events"][0]["status"] == "success"
    # 决策轮 LLM 调用 usage 聚合
    assert out["usage"]["total_tokens"] >= 7


@pytest.mark.asyncio
async def test_order_query_loop(monkeypatch):
    """订单查询：LLM 决策 query_order → route=order。"""
    calls = {"execute": []}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        return {"order": {"order_id": "ORD-1", "status": "PAID"}}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="订单状态如下")
        return _resp(tool_calls=[_tool("query_order", {"order_id": "ORD-1"})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    out = await al.run_decision_loop("查 ORD-1", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "order"
    assert calls["execute"][0][0] == "query_order"


@pytest.mark.asyncio
async def test_no_tool_direct_reply(monkeypatch):
    """无工具直接作答：LLM 认为无需工具 → 按意图兜底路由 + direct_reply 透出。"""

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        return _resp(content="订单已发货，预计明天送达。")

    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    out = await al.run_decision_loop("我订单到哪了", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "order"
    assert out["tool_results"] == {}
    assert "发货" in out["direct_reply"]


@pytest.mark.asyncio
async def test_side_effect_guard_routes_business(monkeypatch):
    """副作用工具决策：护栏拦截，route=business，工具不执行、不透出事件。"""
    calls = {"execute": 0}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"] += 1
        raise AssertionError("副作用工具不应被执行")

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        return _resp(tool_calls=[_tool("create_return_order", {"order_id": "ORD-1", "items": ["SKU-1"]})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    out = await al.run_decision_loop("我要退货 ORD-1", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "business"
    assert calls["execute"] == 0
    assert out["tool_events"] == []
    # 透出被拦工具名与参数：orchestrator 据此重映射真实业务意图（否则 business_flow
    # 对非 FLOWS 意图 KeyError）
    assert out["blocked_tool"] == "create_return_order"
    assert out["blocked_args"] == {"order_id": "ORD-1", "items": ["SKU-1"]}


@pytest.mark.asyncio
async def test_query_order_missing_id_override(monkeypatch):
    """query_order 缺 order_id：override 为 list_user_orders 执行（复用无单号兜底语义）。"""
    calls = {"execute": []}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        return {"orders": [{"order_id": "ORD-1", "status": "SHIPPED"}]}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="以下是您的订单")
        return _resp(tool_calls=[_tool("query_order", {})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    out = await al.run_decision_loop("我有哪些订单", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "order"
    assert calls["execute"][0][0] == "list_user_orders"
    assert out["tool_events"][0]["name"] == "list_user_orders"  # 事件名用 override 后工具名


@pytest.mark.asyncio
async def test_max_rounds_exhausted_routes_by_intent(monkeypatch):
    """循环轮次耗尽（LLM 每轮调不同参数的工具且一直有 tool_calls）：按工具结果路由。

    用不同 order_id 避开 P4 dedupe（同参数二次调用不执行），确保真正测到轮次截断。
    """
    calls = {"execute": [], "round": 0}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append(params.get("order_id"))
        return {"order": {"order_id": params.get("order_id"), "status": "PAID"}}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        calls["round"] += 1
        oid = "ORD-1" if calls["round"] == 1 else "ORD-2"
        return _resp(tool_calls=[_tool("query_order", {"order_id": oid})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    monkeypatch.setattr(al.settings, "agent_loop_max_rounds", 2)
    out = await al.run_decision_loop("查订单", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "order"  # 跑满轮数按已得工具结果路由
    assert calls["execute"] == ["ORD-1", "ORD-2"]  # 两轮不同参数各执行一次
    assert out["tool_results"]["query_order"]["order"]["order_id"] == "ORD-2"


@pytest.mark.asyncio
async def test_duplicate_tool_call_deduped(monkeypatch):
    """P4 dedupe：同轮同工具同参数二次决策 → 复用首次结果，不重复执行、不透出重复事件。"""
    calls = {"execute": []}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        return {"order": {"order_id": "ORD-1", "status": "PAID"}}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="订单状态如下")
        return _resp(tool_calls=[_tool("query_order", {"order_id": "ORD-1"})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    monkeypatch.setattr(al.settings, "agent_loop_max_rounds", 3)
    out = await al.run_decision_loop("查 ORD-1", "ORDER_STATUS", _Session(), 1)
    assert calls["execute"] == [("query_order", {"order_id": "ORD-1"})]  # 同参数只执行一次
    assert len(out["tool_events"]) == 1  # 缓存命中不透出重复事件
    assert out["tool_results"]["query_order"]["order"]["order_id"] == "ORD-1"


@pytest.mark.asyncio
async def test_decision_loop_passes_deepseek_tools_direct(monkeypatch):
    """决策循环直接透传 registry 的 TOOL_SCHEMAS（DeepSeek 格式），调用点无运行时包装。"""
    received = {"tools": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        received["tools"] = tools
        return _resp(content="直接作答")

    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    await al.run_decision_loop("退货政策", "POLICY_INQUIRY", _Session(), 1)
    assert received["tools"] is al.TOOL_SCHEMAS  # 同一对象直传，无二次包装
    assert received["tools"][0]["type"] == "function"
    assert received["tools"][0]["function"]["name"] == "query_order"


@pytest.mark.asyncio
async def test_force_policy_search_gate(monkeypatch):
    """回退闸门：FORCE_POLICY_SEARCH=True 且决策未检索 → 强制补一次 search_policy。"""
    calls = {"execute": []}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        return {"results": [{"text": "政策文档", "score": 0.9, "source": "x.md"}]}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        return _resp(content="直接作答")

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    monkeypatch.setattr(al.settings, "agent_loop_force_policy_search", True)
    out = await al.run_decision_loop("退货政策", "POLICY_INQUIRY", _Session(), 1)
    assert "search_policy" in out["tool_results"]
    assert calls["execute"][0][0] == "search_policy"


@pytest.mark.asyncio
async def test_trivial_policy_query_rejected(monkeypatch):
    """P4 规则1：search_policy 过短/纯问候 → 软拒绝跳过执行（不透出事件），LLM 改写后执行。"""
    calls = {"execute": 0, "round": 0}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"] += 1
        return {"results": [{"text": "政策文档", "score": 0.9, "source": "x.md"}]}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if calls["round"] == 0:
            calls["round"] += 1
            return _resp(tool_calls=[_tool("search_policy", {"query": "你好"}, cid="c1")])
        # 第二轮：LLM 看到被拒占位结果后改写为完整查询
        return _resp(tool_calls=[_tool("search_policy", {"query": "退货政策是什么"}, cid="c2")])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    out = await al.run_decision_loop("退货政策", "POLICY_INQUIRY", _Session(), 1)
    assert calls["execute"] == 1  # 首轮过短被拒不执行，次轮改写后执行一次
    assert out["route"] == "policy"
    assert len(out["tool_events"]) == 1  # 被拒调用不透出事件，只透实际执行
    assert out["tool_events"][0]["name"] == "search_policy"


@pytest.mark.asyncio
async def test_call_limit_truncates_loop(monkeypatch):
    """P4 规则6：累计工具调用达上限 → 截断强制出路由，超限工具不执行。"""
    calls = {"execute": [], "round": 0}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append(params.get("order_id"))
        return {"order": {"order_id": params.get("order_id"), "status": "PAID"}}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if calls["round"] == 0:
            calls["round"] += 1
            return _resp(tool_calls=[_tool("query_order", {"order_id": "ORD-1"}, cid="c1"),
                                     _tool("query_order", {"order_id": "ORD-2"}, cid="c2")])
        return _resp(tool_calls=[_tool("query_order", {"order_id": "ORD-3"}, cid="c3"),
                                 _tool("query_order", {"order_id": "ORD-4"}, cid="c4")])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    monkeypatch.setattr(al.settings, "agent_loop_max_rounds", 2)
    out = await al.run_decision_loop("查订单", "ORDER_STATUS", _Session(), 1)
    assert calls["execute"] == ["ORD-1", "ORD-2", "ORD-3"]  # 第 4 次调用被护栏截断
    assert out["route"] == "order"
    assert len(out["tool_events"]) == 3


@pytest.mark.asyncio
async def test_call_limit_truncates_before_final_round(monkeypatch):
    """P4 规则6：默认 max_rounds=3 下提前达上限 → 截断并终止 LLM 轮次循环。

    回归 hook 缺陷：截断只 break 内层 call 循环会漏掉外层轮次，下一轮仍发 chat
    （携带未闭环 tool_call 引用）且"强制出路由"落空。应恰好 1 次 chat。
    """
    calls = {"execute": [], "chat": 0}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append(params.get("order_id"))
        return {"order": {"order_id": params.get("order_id"), "status": "PAID"}}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        calls["chat"] += 1
        return _resp(tool_calls=[
            _tool("query_order", {"order_id": "ORD-1"}, cid="c1"),
            _tool("query_order", {"order_id": "ORD-2"}, cid="c2"),
            _tool("query_order", {"order_id": "ORD-3"}, cid="c3"),
            _tool("query_order", {"order_id": "ORD-4"}, cid="c4"),
        ])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.deepseek_client, "chat", fake_chat)
    out = await al.run_decision_loop("查订单", "ORDER_STATUS", _Session(), 1)  # 默认 max_rounds=3
    assert calls["execute"] == ["ORD-1", "ORD-2", "ORD-3"]  # 第 4 个被截断
    assert calls["chat"] == 1  # 截断后不再发下一轮 LLM 请求（外层循环被终止）
    assert out["route"] == "order"
