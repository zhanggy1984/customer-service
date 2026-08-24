"""agent_loop：LLM 工具决策循环（P3）。

对 ORDER_STATUS / POLICY_INQUIRY 两类意图运行 LLM 工具决策循环：
LLM 自主决定调用哪些工具（TOOL_SCHEMAS 全量可见），循环内只执行只读白名单工具，
副作用工具决策被护栏拦截并路由到 business_flow 状态机（确定性接手，防 LLM 乱调）。
返回 route + tool_results + tool_events + usage，由 orchestrator 节点 emit 事件并聚合 usage。

设计取舍（对 fc-plan 的收敛，非偏离）：
- 业务意图 / CHITCHAT 不跑本循环：状态机是契约钉死的确定性权威，LLM 干扰破坏
  "订单已校验→资格已查→用户已确认"顺序保证；闲聊无工具可调。短路在 orchestrator 节点。
- 循环 ≤ AGENT_LOOP_MAX_ROUNDS 轮，超限按 intent 兜底路由。
- 回退闸门 AGENT_LOOP_FORCE_POLICY_SEARCH：policy 意图强制补一次 search_policy
  （评测扣分时一键回退，默认关闭）。
- 决策轮 LLM 调用 usage 由调用方聚合进本轮总 usage（契约 §5.1）。
"""
import json
import time

from app.agent.function_calling.executor import execute
from app.agent.function_calling.registry import TOOL_SCHEMAS
from app.config import settings
from app.infrastructure.deepseek import (
    AllKeysDownError,
    CapacityExceededError,
    LLMUnavailableError,
    deepseek_client,
)
from app.utils.logger import logger

LLM_FALLBACK_ERRORS = (LLMUnavailableError, CapacityExceededError, AllKeysDownError)

# 决策循环内只执行只读白名单工具；其余工具决策由护栏拦截
READONLY_TOOLS = {"query_order", "list_user_orders", "search_policy"}
# 副作用工具：LLM 决策到这些工具 → 不执行、不透出，route=business（状态机确定性接手）
SIDE_EFFECT_TOOLS = {
    "check_return_eligibility",
    "create_return_order",
    "create_refund_order",
    "create_complaint",
}

DECISION_PROMPT = (
    "你是电商客服助手，负责决定是否调用工具来回答用户问题。\n"
    "可用工具：{tools}。\n"
    "调用规则：\n"
    "1. 政策/规则/售后 FAQ 类问题（退货、退款、投诉政策）必须调用 search_policy 获取文档依据，勿凭常识作答；\n"
    "2. 订单状态/详情查询应调用 query_order（需订单号）或 list_user_orders（列最近订单）；\n"
    "3. 只有当你已通过工具拿到足够信息，才停止调用工具，并基于工具结果回答用户；\n"
    "4. 工具结果不足或无法确定时，再调用一个更合适的工具，不要直接臆断作答。"
)

_EMPTY_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 0,
}


def _merge_usage(acc: dict, usage: dict | None) -> dict:
    """累加一次 chat 的 usage（None 安全跳过）。"""
    if not usage:
        return acc
    for k in _EMPTY_USAGE:
        acc[k] += usage.get(k, 0) or 0
    return acc


async def _execute_guarded(
    name: str,
    params: dict,
    user_id: int,
    session_id: str,
    tool_events: list,
) -> dict | None:
    """护栏包装执行。返回结果 dict；副作用工具被拦返回 None（调用方据此 route=business）。"""
    if name in SIDE_EFFECT_TOOLS:
        logger.info("event=fc_guard_side_effect", extra={"tool": name})
        return None
    if name == "query_order" and not (params.get("order_id") or "").strip():
        # 无订单号 → 兜底列最近订单（沿用既有"无单号列订单辅助定位"语义）
        name = "list_user_orders"
        params = {"limit": 5}
        logger.info("event=fc_guard_override", extra={"from": "query_order", "to": "list_user_orders"})
    result = await execute(name, params, user_id, session_id)
    tool_events.append({
        "id": str(time.time_ns()),
        "name": name,
        "args": params,
        "result": result,
        "status": "error" if result.get("error") else "success",
    })
    return result


def _decide_route(intent: str, tool_results: dict) -> str:
    """无 tool_calls 出循环后的路由判定：按工具结果归属，无结果按 intent 兜底。"""
    if "search_policy" in tool_results:
        return "policy"
    if any(t in tool_results for t in ("query_order", "list_user_orders")):
        return "order"
    return "policy" if intent == "POLICY_INQUIRY" else "order"


async def run_decision_loop(user_message: str, intent: str, session, user_id: int) -> dict:
    """LLM 工具决策循环（仅 ORDER_STATUS / POLICY_INQUIRY；其他意图由 orchestrator 短路）。

    返回：
    - route: "order" | "policy" | "business"（生成节点路由；business=护栏拦截副作用工具决策）
    - tool_results: {tool_name: result}（供生成节点注入组装回复）
    - tool_events: [{"id","name","args","result","status"}]（由调用方 emit，观测层透出）
    - usage: 决策轮 LLM 调用聚合 token（由调用方 usage.accumulate 计入本轮）
    - direct_reply: LLM 无工具调用直接作答的 content（生成节点透出，纯自主语义）
    """
    # 防御：非决策类意图直接短路（正常路径由 orchestrator 短路，不达此处）
    if intent not in ("ORDER_STATUS", "POLICY_INQUIRY"):
        return {"route": "business", "tool_results": {}, "tool_events": [],
                "usage": dict(_EMPTY_USAGE), "direct_reply": ""}

    session_id = getattr(session, "session_id", "")
    tool_results: dict = {}
    tool_events: list = []
    decision_usage = dict(_EMPTY_USAGE)
    direct_reply = ""  # LLM 无工具调用直接作答的 content（纯自主语义，生成节点透出）

    messages = [
        {"role": "system", "content": DECISION_PROMPT.format(
            tools=json.dumps([t["function"]["name"] for t in TOOL_SCHEMAS], ensure_ascii=False))},
        {"role": "user", "content": user_message},
    ]

    for _ in range(settings.agent_loop_max_rounds):
        try:
            data = await deepseek_client.chat(
                messages,
                model=settings.deepseek_model_chat,
                temperature=0.1,  # 工具决策需确定性
                tools=TOOL_SCHEMAS,  # registry 即 DeepSeek 传输格式（type=function 包装）
                tool_choice="auto",
            )
        except LLM_FALLBACK_ERRORS:
            raise  # LLM 熔断 → 冒泡给 _rule_engine_fallback 装饰器统一降级
        except Exception as exc:  # 非 LLM 异常（解析失败等）：退出循环按现有结果路由
            logger.error("event=agent_loop_error", extra={"error": str(exc)})
            break
        decision_usage = _merge_usage(decision_usage, data.get("usage"))

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            direct_reply = content  # 纯自主：LLM 认为无需工具直接作答，透出给生成节点
            break  # 不再调工具 → 出循环定路由

        # 回灌本轮 assistant 消息（含 tool_calls），供下一轮 LLM 理解已决策动作
        messages.append({"role": "assistant", "content": content or None, "tool_calls": tool_calls})

        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            try:
                params = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                params = {}
            result = await _execute_guarded(name, params, user_id, session_id, tool_events)
            # 副作用工具决策 → 护栏拦截，route=business（状态机确定性接手）。丢弃已读结果，
            # 但透出被拦工具名与参数：orchestrator 据此重映射真实业务意图（如 ORDER_STATUS
            # 误分类 + create_return_order → RETURN_REQUEST），否则 business_flow 索引 FLOWS
            # 对非业务意图抛 KeyError。
            if result is None:
                return {"route": "business", "blocked_tool": name, "blocked_args": params,
                        "tool_results": tool_results, "tool_events": tool_events,
                        "usage": decision_usage, "direct_reply": ""}
            tool_results[name] = result
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": json.dumps(result, ensure_ascii=False),
            })

    # 回退闸门：policy 意图强制补一次 search_policy（评测扣分时一键回退）
    if (intent == "POLICY_INQUIRY" and settings.agent_loop_force_policy_search
            and "search_policy" not in tool_results):
        result = await execute("search_policy", {"query": user_message}, user_id, session_id)
        tool_results["search_policy"] = result
        tool_events.append({
            "id": str(time.time_ns()), "name": "search_policy",
            "args": {"query": user_message[:50]}, "result": result,
            "status": "error" if result.get("error") else "success",
        })
        logger.info("event=fc_force_policy_search")

    route = _decide_route(intent, tool_results)
    return {"route": route, "tool_results": tool_results, "tool_events": tool_events,
            "usage": decision_usage, "direct_reply": direct_reply}
