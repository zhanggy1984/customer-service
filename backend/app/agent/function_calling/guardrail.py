"""实时护栏 ToolGuardrail（P4）：决策与执行之间的确定性规则校验。

P3 的最小护栏内联在 agent_loop（只读白名单 + query_order 缺 id override + 副作用工具
reject→business）。P4 升级为独立校验器：输出 allow/reject/override 三态 + 理由，新增
search_policy 过短/纯问候 reject、同轮同工具同参数 dedupe、累计调用超限截断。
判定理由经 logger 记 event=guardrail_verdict（tool_call_log 表 P5 建，此处不落库）。

状态化实例 per 决策循环：dedupe 缓存与调用计数随实例生命周期，跨轮有效。
规则短路顺序：副作用 → 截断 → dedupe → 内容规则 → override → allow。
"""
import json
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["allow", "reject", "override"]


@dataclass
class GuardDecision:
    verdict: Verdict
    reason: str = ""  # 机器可读判定理由，P5 落 tool_call_log 字段
    tool_name: str | None = None  # override 目标工具
    params: dict | None = None  # override 参数
    cached_result: dict | None = None  # dedupe 命中的首次执行结果


# 副作用工具：LLM 在决策轮决策到这些工具 → 不执行、reject，路由 business_flow 状态机接手
SIDE_EFFECT_TOOLS = {
    "check_return_eligibility",
    "create_return_order",
    "create_refund_order",
    "create_complaint",
}
# 纯问候/寒暄词：search_policy 对纯寒暄无意义（决策 prompt 已引导政策问题才检索，此为防御）
_GREETING_WORDS = ("你好", "您好", "hi", "hello", "在吗", "在么", "哈喽", "早上好")
# 单轮决策循环累计工具调用上限（fc-plan §P4 文档既定）。注意与 AGENT_LOOP_MAX_ROUNDS
# （LLM 调用轮数上限）不同维度：此处限实际执行工具总数，防御 LLM 每轮多 call 刷屏。
MAX_TOOL_CALLS = 3


def _is_greeting(query: str) -> bool:
    """纯问候判定：去标点后整句等于问候词（避免"你好，退货政策是什么"被误伤）。"""
    return query.strip().strip("？！?!。，,.") in _GREETING_WORDS


class ToolGuardrail:
    """决策与执行之间的确定性规则校验器。per 决策循环实例化。"""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], dict] = {}  # (工具, 参数 json) → 首次结果
        self._call_count = 0  # 已实际执行的工具调用数

    def check(self, tool_name: str, params: dict) -> GuardDecision:
        """按规则短路判定，返回 allow/reject/override + 理由。"""
        # 1. 副作用工具 → reject：LLM 不该在决策轮发起业务动作，状态机确定性接手
        if tool_name in SIDE_EFFECT_TOOLS:
            return GuardDecision("reject", "side_effect")
        # 2. 累计调用超限 → reject：防御 LLM 刷工具调用，强制出路由（不执行新工具）
        if self._call_count >= MAX_TOOL_CALLS:
            return GuardDecision("reject", "too_many_calls")
        # 3. 同轮同工具同参数 → dedupe：复用首次结果，不重复执行（LLM 多轮循环可能重复决策）
        key = (tool_name, json.dumps(params, sort_keys=True, ensure_ascii=False))
        if key in self._seen:
            return GuardDecision("allow", "dedupe", cached_result=self._seen[key])
        # 4. search_policy 过短/纯问候 → reject：检索无意义，LLM 应改写查询或放弃调工具
        if tool_name == "search_policy":
            query = (params.get("query") or "").strip()
            if len(query) < 4 or _is_greeting(query):
                return GuardDecision("reject", "trivial_query")
        # 5. query_order 无 order_id → override：降级为列最近订单辅助定位（无单号兜底语义）。
        #    先对 override 目标查 dedupe——否则 dedupe 缓存键是原始 (query_order,{})，override
        #    目标 (list_user_orders,{limit:5}) 永不命中，同参数二次决策会重复执行，违反 dedupe 规则。
        if tool_name == "query_order" and not (params.get("order_id") or "").strip():
            target_key = ("list_user_orders", json.dumps({"limit": 5}, sort_keys=True, ensure_ascii=False))
            if target_key in self._seen:
                return GuardDecision("allow", "dedupe", cached_result=self._seen[target_key])
            return GuardDecision("override", "missing_order_id",
                                 tool_name="list_user_orders", params={"limit": 5})
        return GuardDecision("allow")

    def record(self, tool_name: str, params: dict, result: dict) -> None:
        """实际执行后登记：dedupe 缓存 + 调用计数（仅真执行累加，reject/dedupe 不计数）。"""
        self._call_count += 1
        self._seen[(tool_name, json.dumps(params, sort_keys=True, ensure_ascii=False))] = result
