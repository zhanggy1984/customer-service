"""FC 工具 Schema：供 LLM 选择工具时使用。

TOOL_SCHEMAS 即 DeepSeek（OpenAI 兼容）的 tools 参数格式：每个元素 `{"type": "function",
"function": {name, description, parameters}}`。此前为 name/description/parameters 平铺格式，
调用点须运行时包装（DeepSeek 对缺 type 的 tools 返回 400 missing field `type`），易忘。
现统一为传输格式，加新工具只改此处一处，LLM 调用点直接透传。

契约规范（FC 契约优化，参考 good-question RETRIEVE_TOOL_SCHEMA）：
- 所有 parameters 显式 additionalProperties=False，防 LLM 塞未声明参数；
- 参数 description 补齐（缺失时 LLM 只能靠函数名猜参数语义）；
- search_policy 的 description 文档化返回结构 + query 构造指导（query 清洗源头闸门）。
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
                "additionalProperties": False,
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
                "additionalProperties": False,
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
                "properties": {"order_id": {"type": "string", "description": "订单号，如 ORD-20240801-001"}},
                "required": ["order_id"],
                "additionalProperties": False,
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
                    "order_id": {"type": "string", "description": "订单号，如 ORD-20240801-001"},
                    "items": {"type": "array", "items": {"type": "string"}, "description": "商品 SKU 列表"},
                    "reason": {"type": "string", "description": "退货原因"},
                },
                "required": ["order_id", "items"],
                "additionalProperties": False,
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
                    "order_id": {"type": "string", "description": "订单号，如 ORD-20240801-001"},
                    "reason": {"type": "string", "description": "退款原因"},
                },
                "required": ["order_id"],
                "additionalProperties": False,
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
                    "complaint_type": {"type": "string", "description": "投诉类型，如商品质量/物流/服务"},
                    "description": {"type": "string", "description": "投诉描述"},
                    "order_id": {"type": "string", "description": "关联订单号（可选）"},
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "检索客服政策知识库（退货/退款/售后政策、FAQ）。\n"
                "query 取用户问题的核心实体与关键限制条件，去除寒暄客套，不要照抄整段对话，通常 1-2 句。\n"
                "返回 JSON：data.results（命中片段列表，每项含 text/score/source）、data.source_count（命中条数）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询词，取核心实体与限制条件，1-2 句"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]
