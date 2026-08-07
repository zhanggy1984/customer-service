"""FC 工具 Schema：供 LLM 选择工具时使用（Agent Pipeline 阶段 5 注入）。"""

TOOL_SCHEMAS = [
    {
        "name": "query_order",
        "description": "按订单号查询订单详情（含商品明细、状态）。订单必须属于当前用户。",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号，如 ORD-20240801-001"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "list_user_orders",
        "description": "列出当前用户最近的订单列表。",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "数量，默认 5"}},
            "required": [],
        },
    },
    {
        "name": "check_return_eligibility",
        "description": "检查订单退货资格：是否可退、可退金额、可退商品清单。",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
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
    {
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
    {
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
    {
        "name": "search_policy",
        "description": "检索客服政策知识库（退货/退款/售后政策、FAQ）。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_SCHEMAS]
