"""工具 handler 注册表。每个 handler 签名统一: (params: dict, user_id: int, session_id: str) -> dict"""
from app.agent.function_calling.tools import (
    complaint_tools,
    order_tools,
    policy_tools,
    refund_tools,
    return_tools,
)

HANDLERS = {
    "query_order": order_tools.query_order,
    "list_user_orders": order_tools.list_user_orders,
    "check_return_eligibility": return_tools.check_return_eligibility,
    "create_return_order": return_tools.create_return_order,
    "create_refund_order": refund_tools.create_refund_order,
    "create_complaint": complaint_tools.create_complaint,
    "search_policy": policy_tools.search_policy,
}
