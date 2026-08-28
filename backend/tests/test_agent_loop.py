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


def _resp(content: str = "", tool_calls: list | None = None, reasoning: str = "") -> dict:
    msg: dict = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning:
        msg["reasoning_content"] = reasoning  # thinking 开启时非流式响应携带思考链
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
        return {"ok": True, "data": {"results": [{"text": "签收后 7 天内支持无理由退货。", "score": 0.9, "source": "x.md"}]}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="基于政策文档回答...")
        return _resp(tool_calls=[_tool("search_policy", {"query": "退货政策是什么？"})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
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
        return {"ok": True, "data": {"order": {"order_id": "ORD-1", "status": "PAID"}}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="订单状态如下")
        return _resp(tool_calls=[_tool("query_order", {"order_id": "ORD-1"})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    # 输入不带订单号（避免触发优化③规则短路），保持"LLM 决策 query_order"语义
    out = await al.run_decision_loop("查询我的订单状态", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "order"
    assert calls["execute"][0][0] == "query_order"


@pytest.mark.asyncio
async def test_no_tool_direct_reply(monkeypatch):
    """无工具直接作答：LLM 认为无需工具 → 按意图兜底路由 + direct_reply 透出。"""

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        return _resp(content="订单已发货，预计明天送达。")

    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    out = await al.run_decision_loop("我订单到哪了", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "order"
    assert out["tool_results"] == {}
    assert "发货" in out["direct_reply"]


@pytest.mark.asyncio
async def test_decision_reasoning_aggregated(monkeypatch):
    """thinking 透出：多轮决策的 reasoning_content 按轮次聚合为 reasoning 字段。

    决策循环开启 thinking 后，各轮 chat 的 message.reasoning_content 携带思考链；
    循环内逐轮累积，最终以换行 join 透出（direct_reply 无工具作答轮同样携带，
    生成节点据此在决策直接回答时也展示思考过程）。
    """
    calls = {"round": 0}

    async def fake_execute(name, params, user_id, session_id):
        return {"ok": True, "data": {"results": [{"text": "政策文档", "score": 0.9, "source": "x.md"}]}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if calls["round"] == 0:
            calls["round"] += 1
            # 首轮：思考后决定调 search_policy
            return _resp(tool_calls=[_tool("search_policy", {"query": "退货政策"})],
                         reasoning="用户问退货政策，属政策/FAQ 类，须检索政策文档。")
        # 次轮：拿到工具结果后直接作答（含第二段思考）
        return _resp(content="基于政策文档回答...", reasoning="工具结果已足够，直接作答。")

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    out = await al.run_decision_loop("退货政策是什么？", "POLICY_INQUIRY", _Session(), 1)
    assert out["route"] == "policy"
    assert "search_policy" in out["tool_results"]
    # 两轮思考按轮次顺序 join（非覆盖，非乱序）
    assert out["reasoning"] == "用户问退货政策，属政策/FAQ 类，须检索政策文档。\n工具结果已足够，直接作答。"


@pytest.mark.asyncio
async def test_decision_direct_reply_reasoning(monkeypatch):
    """无工具直接作答路径：单轮 LLM 思考 + 作答，reasoning 透出（决策直接回答同样可见思考）。"""

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        return _resp(content="订单已发货，预计明天送达。", reasoning="用户问物流进度，无工具可查，直接告知。")

    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    out = await al.run_decision_loop("我订单到哪了", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "order"
    assert out["direct_reply"] == "订单已发货，预计明天送达。"
    assert out["reasoning"] == "用户问物流进度，无工具可查，直接告知。"


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
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    # 输入不带订单号（避免触发规则短路），保持"LLM 决策副作用工具 → 护栏拦截"语义
    out = await al.run_decision_loop("我要退货", "ORDER_STATUS", _Session(), 1)
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
        return {"ok": True, "data": {"orders": [{"order_id": "ORD-1", "status": "SHIPPED"}]}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="以下是您的订单")
        return _resp(tool_calls=[_tool("query_order", {})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
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
        return {"ok": True, "data": {"order": {"order_id": params.get("order_id"), "status": "PAID"}}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        calls["round"] += 1
        oid = "ORD-1" if calls["round"] == 1 else "ORD-2"
        return _resp(tool_calls=[_tool("query_order", {"order_id": oid})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    monkeypatch.setattr(al.settings, "agent_loop_max_rounds", 2)
    out = await al.run_decision_loop("查订单", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "order"  # 跑满轮数按已得工具结果路由
    assert calls["execute"] == ["ORD-1", "ORD-2"]  # 两轮不同参数各执行一次
    assert out["tool_results"]["query_order"]["data"]["order"]["order_id"] == "ORD-2"


@pytest.mark.asyncio
async def test_duplicate_tool_call_deduped(monkeypatch):
    """P4 dedupe：同轮同工具同参数二次决策 → 复用首次结果，不重复执行、不透出重复事件。"""
    calls = {"execute": []}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        return {"ok": True, "data": {"order": {"order_id": "ORD-1", "status": "PAID"}}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="订单状态如下")
        return _resp(tool_calls=[_tool("query_order", {"order_id": "ORD-1"})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    monkeypatch.setattr(al.settings, "agent_loop_max_rounds", 3)
    # 输入不带订单号（避免触发规则短路），保持"LLM 重复决策 → dedupe"语义
    out = await al.run_decision_loop("查询订单详情", "ORDER_STATUS", _Session(), 1)
    assert calls["execute"] == [("query_order", {"order_id": "ORD-1"})]  # 同参数只执行一次
    assert len(out["tool_events"]) == 1  # 缓存命中不透出重复事件
    assert out["tool_results"]["query_order"]["data"]["order"]["order_id"] == "ORD-1"


@pytest.mark.asyncio
async def test_decision_loop_passes_deepseek_tools_direct(monkeypatch):
    """决策循环直接透传 registry 的 TOOL_SCHEMAS（DeepSeek 格式），调用点无运行时包装。"""
    received = {"tools": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        received["tools"] = tools
        return _resp(content="直接作答")

    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
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
        return {"ok": True, "data": {"results": [{"text": "政策文档", "score": 0.9, "source": "x.md"}]}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        return _resp(content="直接作答")

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
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
        return {"ok": True, "data": {"results": [{"text": "政策文档", "score": 0.9, "source": "x.md"}]}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if calls["round"] == 0:
            calls["round"] += 1
            return _resp(tool_calls=[_tool("search_policy", {"query": "你好"}, cid="c1")])
        # 第二轮：LLM 看到被拒占位结果后改写为完整查询
        return _resp(tool_calls=[_tool("search_policy", {"query": "退货政策是什么"}, cid="c2")])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
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
        return {"ok": True, "data": {"order": {"order_id": params.get("order_id"), "status": "PAID"}}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if calls["round"] == 0:
            calls["round"] += 1
            return _resp(tool_calls=[_tool("query_order", {"order_id": "ORD-1"}, cid="c1"),
                                     _tool("query_order", {"order_id": "ORD-2"}, cid="c2")])
        return _resp(tool_calls=[_tool("query_order", {"order_id": "ORD-3"}, cid="c3"),
                                 _tool("query_order", {"order_id": "ORD-4"}, cid="c4")])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    monkeypatch.setattr(al.settings, "agent_loop_max_rounds", 2)
    out = await al.run_decision_loop("查订单", "ORDER_STATUS", _Session(), 1)
    assert calls["execute"] == ["ORD-1", "ORD-2", "ORD-3"]  # 第 4 次调用被护栏截断
    assert out["route"] == "order"
    assert len(out["tool_events"]) == 3


@pytest.mark.asyncio
async def test_decision_loop_writes_tool_call_log(monkeypatch):
    """P5 落库接入：决策循环真执行工具后应写 tool_call_log（收集 write_tool_call 调用）。"""
    calls = []

    async def fake_log(**kwargs):
        calls.append(kwargs)

    async def fake_execute(name, params, user_id, session_id):
        return {"ok": True, "data": {"order": {"order_id": "ORD-1", "status": "PAID"}}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="订单状态如下")
        return _resp(tool_calls=[_tool("query_order", {"order_id": "ORD-1"})])

    monkeypatch.setattr(al, "write_tool_call", fake_log)  # 覆盖 conftest 的 no-op，改为收集
    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    # 输入不带订单号（避免触发规则短路），保持"LLM 循环真执行 → 落库"语义
    out = await al.run_decision_loop("查询订单详情", "ORDER_STATUS", _Session(), 1)
    assert out["route"] == "order"
    assert len(calls) == 1  # query_order 真执行 → 恰好落 1 条
    entry = calls[0]
    assert entry["tool_name"] == "query_order"
    assert entry["verdict"] == "allow"
    assert entry["round_no"] == 1
    assert entry["session_id"] == "sess-1"
    assert entry["query_text"] == "查询订单详情"


@pytest.mark.asyncio
async def test_call_limit_truncates_before_final_round(monkeypatch):
    """P4 规则6：默认 max_rounds=3 下提前达上限 → 截断并终止 LLM 轮次循环。

    回归 hook 缺陷：截断只 break 内层 call 循环会漏掉外层轮次，下一轮仍发 chat
    （携带未闭环 tool_call 引用）且"强制出路由"落空。应恰好 1 次 chat。
    """
    calls = {"execute": [], "chat": 0}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append(params.get("order_id"))
        return {"ok": True, "data": {"order": {"order_id": params.get("order_id"), "status": "PAID"}}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        calls["chat"] += 1
        return _resp(tool_calls=[
            _tool("query_order", {"order_id": "ORD-1"}, cid="c1"),
            _tool("query_order", {"order_id": "ORD-2"}, cid="c2"),
            _tool("query_order", {"order_id": "ORD-3"}, cid="c3"),
            _tool("query_order", {"order_id": "ORD-4"}, cid="c4"),
        ])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    out = await al.run_decision_loop("查订单", "ORDER_STATUS", _Session(), 1)  # 默认 max_rounds=3
    assert calls["execute"] == ["ORD-1", "ORD-2", "ORD-3"]  # 第 4 个被截断
    assert calls["chat"] == 1  # 截断后不再发下一轮 LLM 请求（外层循环被终止）
    assert out["route"] == "order"


# ==================== 优化③：决策循环规则短路（ORDER_STATUS 半短路） ====================

@pytest.mark.asyncio
async def test_rule_shortcut_order_status_hit(monkeypatch):
    """规则短路命中：ORDER_STATUS + 订单号 → 确定性直查 query_order，零 LLM 调用。

    产出与 LLM 决策循环契约同构（route=order + tool_results + tool_events），
    落库 verdict=rule_shortcut（非 allow），观测可区分短路与 LLM 决策路径。
    """
    calls = {"execute": [], "chat": 0, "log": []}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        return {"ok": True, "data": {"order": {"order_id": "ORD-20240801-001", "status": "PAID"}}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        calls["chat"] += 1
        raise AssertionError("短路命中不应调用 LLM 决策")

    async def fake_log(**kwargs):
        calls["log"].append(kwargs)

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    monkeypatch.setattr(al, "write_tool_call", fake_log)
    out = await al.run_decision_loop("查 ORD-20240801-001", "ORDER_STATUS", _Session(), 1)
    assert calls["chat"] == 0  # 决策轮 LLM 零调用
    assert calls["execute"] == [("query_order", {"order_id": "ORD-20240801-001"})]
    assert out["route"] == "order"
    assert out["tool_results"]["query_order"]["data"]["order"]["order_id"] == "ORD-20240801-001"
    assert out["tool_events"][0]["name"] == "query_order"
    assert out["direct_reply"] == ""
    assert len(calls["log"]) == 1
    assert calls["log"][0]["verdict"] == "rule_shortcut"
    assert calls["log"][0]["tool_name"] == "query_order"
    assert calls["log"][0]["query_text"] == "查 ORD-20240801-001"


@pytest.mark.asyncio
async def test_rule_shortcut_not_found_chains_list(monkeypatch):
    """短路连查：query_order 未命中订单 → 顺序连查 list_user_orders 兜底。

    等价 LLM 决策循环典型多步路径，但确定性一步完成；生成节点 _compose_order_answer
    已支持"query_order 未命中 + list_user_orders 命中 → 列最近订单"组装。
    """
    calls = {"execute": [], "log": []}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        if name == "query_order":
            return {"ok": False, "data": None,
                    "error": {"code": "order_not_found", "message": "订单不存在"}}
        return {"ok": True, "data": {"orders": [{"order_id": "ORD-20240801-001", "status": "PAID"}]}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        raise AssertionError("短路命中不应调用 LLM 决策")

    async def fake_log(**kwargs):
        calls["log"].append(kwargs)

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    monkeypatch.setattr(al, "write_tool_call", fake_log)
    out = await al.run_decision_loop("查 ORD-999999999", "ORDER_STATUS", _Session(), 1)
    assert calls["execute"] == [("query_order", {"order_id": "ORD-999999999"}),
                                ("list_user_orders", {"limit": 5})]  # 连查
    assert "query_order" in out["tool_results"] and "list_user_orders" in out["tool_results"]
    assert len(out["tool_events"]) == 2
    assert len(calls["log"]) == 2  # 两次调用各落一条，均为 rule_shortcut
    assert {e["reason"] for e in calls["log"]} == {"rule_shortcut_order_id", "rule_shortcut_fallback"}
    # 连查数据必须对用户可见：生成节点组装出"最近订单"（修复 _compose_order_answer 提前 return）
    from app.agent.orchestrator import _compose_order_answer
    reply = _compose_order_answer(out["tool_results"])
    assert "您最近的订单" in reply
    assert "ORD-20240801-001" in reply


@pytest.mark.asyncio
async def test_rule_shortcut_miss_falls_back_llm(monkeypatch):
    """短路未命中回退 LLM：POLICY_INQUIRY（即使带单号）、ORDER_STATUS 无单号均走决策循环。"""
    calls = {"execute": [], "chat": 0}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        return {"ok": True, "data": {"results": [{"text": "政策文档", "score": 0.9, "source": "x.md"}]}, "error": None}

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        calls["chat"] += 1
        if any(m.get("role") == "tool" for m in messages):
            return _resp(content="基于政策文档回答")
        return _resp(tool_calls=[_tool("search_policy", {"query": "退货政策"})])

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    # POLICY_INQUIRY 带单号 → 不短路（政策侧不接管，防检索 query 改写丢失）
    out = await al.run_decision_loop("ORD-20240801-001 能退吗", "POLICY_INQUIRY", _Session(), 1)
    assert calls["chat"] >= 1  # 走 LLM 决策循环（非短路）
    assert calls["execute"][0][0] == "search_policy"
    # ORDER_STATUS 无单号 → 不短路，走 LLM 决策循环
    calls["chat"] = 0
    calls["execute"] = []
    out = await al.run_decision_loop("我的订单到哪了", "ORDER_STATUS", _Session(), 1)
    assert calls["chat"] >= 1


@pytest.mark.asyncio
async def test_rule_shortcut_exception_falls_back_llm(monkeypatch):
    """短路执行异常（只读查询故障）→ 回退 LLM 决策循环，不阻断本轮。"""
    calls = {"execute": [], "chat": 0}

    async def fake_execute(name, params, user_id, session_id):
        calls["execute"].append((name, params))
        raise RuntimeError("db down")

    async def fake_chat(messages, model=None, timeout=None, temperature=None, tools=None, tool_choice=None):
        calls["chat"] += 1
        return _resp(content="订单查询暂时不可用")

    monkeypatch.setattr(al, "execute", fake_execute)
    monkeypatch.setattr(al.llm_gateway, "chat", fake_chat)
    out = await al.run_decision_loop("查 ORD-20240801-001", "ORDER_STATUS", _Session(), 1)
    assert calls["execute"][0] == ("query_order", {"order_id": "ORD-20240801-001"})  # 短路先执行
    assert calls["chat"] == 1  # 短路异常 → 回退 LLM 决策循环
    assert out["route"] == "order"
    assert out["direct_reply"] == "订单查询暂时不可用"  # LLM 直接作答透出
