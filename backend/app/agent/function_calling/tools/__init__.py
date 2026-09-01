"""工具 handler 注册表。每个 handler 签名统一: (params: dict, user_id: int, session_id: str) -> dict

只注册只读白名单工具（决策循环真正会执行的）。副作用工具（check_return_eligibility /
create_return_order / create_refund_order / create_complaint）不再注册 handler：
- 护栏（guardrail.SIDE_EFFECT_TOOLS）恒 reject 其决策，路由 business_flow 状态机确定性接手，
  handler 体从未被 execute 执行（不可达死代码，且 create_return 的 items 语义是 item_id，
  与 LLM 提取的商品名不匹配，即使放开也必然假拒绝）；
- 它们的契约价值在 TOOL_SCHEMAS（LLM 决策空间）+ SIDE_EFFECT_TOOLS（拦截）+
  _SIDE_EFFECT_TO_INTENT（意图重映射），三条链路均按工具名工作，与 HANDLERS 解耦。
"""
from app.agent.function_calling.tools import (
    order_tools,
    policy_tools,
)

HANDLERS = {
    "query_order": order_tools.query_order,
    "list_user_orders": order_tools.list_user_orders,
    "search_policy": policy_tools.search_policy,
}
