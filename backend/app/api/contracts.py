"""4.3 标准契约清单端点（平台定标准，agent 适配）。

统一 `GET /api/contracts`（公开无鉴权），声明本 agent 的 LLM 评测接口、场景清单与
**驱动契约（contract 段，manifest v2）**。平台脚手架读此端点做接口自动发现（决策
#55/#56）与 adapter 生成（{{input.*}}/{{auth.*}}/{{prepare.*}} 占位符由平台渲染）。
llm=false 为辅助接口（登录等），只登记不进 agent_interface。contract 段是平台驱动本
agent 的权威声明，改动需与平台 seed 快照保持同构（discover 会对比漂移）。
"""
from fastapi import APIRouter

router = APIRouter(prefix="/contracts", tags=["contracts"])

MANIFEST = {
    "agent": "customer-service",
    "contract_version": "2.0",
    "interfaces": [
        {"name": "chat", "path": "/api/v1/sessions/{sid}/messages", "method": "POST",
         "contract_type": "sse", "llm": True,
         "description": "客服会话对话（SSE 流式，透出 token/usage/done；token 事件含 content+delta 双字段，平台 field_map 可映射 answer）"},
        {"name": "login", "path": "/api/v1/auth/login", "method": "POST",
         "llm": False, "description": "会话鉴权（辅助接口）"},
    ],
    "scenes": [
        {"tag": "greeting", "description": "问候与闲聊"},
        {"tag": "order_query", "description": "订单查询"},
        {"tag": "after_sales", "description": "售后服务（退换/退款等）"},
        {"tag": "human_handoff", "description": "转人工客服"},
    ],
    "contract": {
        "type": "sse", "timeout": 120,
        "prepare": [
            {"name": "login", "method": "POST", "path": "/api/v1/auth/login",
             "body": {"username": "{{auth.username}}", "password": "{{auth.password}}"},
             "extract": {"token": "access_token"}},
            # 建 session 返回 {session_id}（非 id）：extract 指定映射
            {"name": "session", "method": "POST", "path": "/api/v1/sessions",
             "headers": {"Authorization": "Bearer {{prepare.login.token}}"},
             "extract": {"id": "session_id"}},
        ],
        "request": {
            "path": "/api/v1/sessions/{{prepare.session.id}}/messages", "method": "POST",
            "headers": {"Authorization": "Bearer {{prepare.login.token}}",
                        "Content-Type": "application/json"},
            "body": {"content": "{{input.content}}"},
        },
    },
}


@router.get("", summary="标准契约清单")
async def contracts() -> dict:
    return MANIFEST
