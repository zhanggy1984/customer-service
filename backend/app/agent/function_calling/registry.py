"""FC 工具 Schema：供 LLM 选择工具时使用。

TOOL_SCHEMAS 即 DeepSeek（OpenAI 兼容）的 tools 参数格式：每个元素 `{"type": "function",
"function": {name, description, parameters}}`。此前为 name/description/parameters 平铺格式，
调用点须运行时包装（DeepSeek 对缺 type 的 tools 返回 400 missing field `type`），易忘。
现统一为传输格式，加新工具只改此处一处，LLM 调用点直接透传。
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "按订单号查询订单详情（含商品明细、状态）。订单必须属于当前用户。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号，如 ORD-20240801-001"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_user_orders",
            "description": "列出当前用户最近的订单列表。",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "数量，默认 5"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_eligibility",
            "description": "检查订单退货资格：是否可退、可退金额、可退商品清单。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return_order",
            "description": "为订单创建退货单。items 为要退的商品 SKU 列表，reason 为退货原因。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "string"}, "description": "商品 SKU 列表"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_refund_order",
            "description": "为订单创建仅退款申请。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_complaint",
            "description": "创建投诉工单。complaint_type 为投诉类型，description 为投诉描述。",
            "parameters": {
                "type": "object",
                "properties": {
                    "complaint_type": {"type": "string"},
                    "description": {"type": "string"},
                    "order_id": {"type": "string"},
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "检索客服政策知识库（退货/退款/售后政策、FAQ）。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]
