"""FC 执行器：根据 tool_name 路由到对应 handler。

统一返回信封 {ok, data, error}（FC 契约）：handler 已返回信封，此处只兜底
未知工具与 handler 异常两类错误，不再二次包装成功结果。
"""
from app.agent.function_calling.tools import HANDLERS
from app.utils.logger import logger


async def execute(tool_name: str, params: dict, user_id: int, session_id: str = "") -> dict:
    """执行工具调用，返回 JSON 友好的信封 dict。"""
    handler = HANDLERS.get(tool_name)
    if not handler:
        logger.warning("event=fc_unknown_tool", extra={"tool": tool_name})
        return {"ok": False, "data": None, "error": {"code": "unknown_tool", "message": f"未知工具 {tool_name}"}}
    logger.info("event=fc_execute", extra={"tool": tool_name, "user_id": user_id})
    try:
        return await handler(params, user_id, session_id)
    except Exception as exc:  # 对接层异常（含 ServiceUnavailableException）
        logger.error("event=fc_error", extra={"tool": tool_name, "error": str(exc)})
        return {"ok": False, "data": None, "error": {"code": "internal_error", "message": "系统出问题了，请稍后重试"}}
