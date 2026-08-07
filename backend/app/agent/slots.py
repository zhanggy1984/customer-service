"""槽位定义与缺失检查。

REQUIRED_SLOTS 定义各意图的必填槽位。missing_slots 非空时，上层不进入状态机，
由 LLM 动态生成追问（同时附带用户最近订单列表）。
"""
from typing import Dict, List

REQUIRED_SLOTS: Dict[str, List[str]] = {
    "POLICY_INQUIRY": [],
    "ORDER_STATUS": ["order_id"],
    "RETURN_REQUEST": ["order_id"],
    "REFUND_REQUEST": ["order_id"],
    "COMPLAINT": [],  # complaint_type 由 LLM 从输入提取，不强制
    "CHITCHAT": [],
}


def check_missing_slots(intent: str, slots: dict) -> list[str]:
    """返回缺失的必填槽位列表。"""
    required = REQUIRED_SLOTS.get(intent, [])
    return [slot for slot in required if not slots.get(slot)]
