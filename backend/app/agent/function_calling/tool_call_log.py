"""tool_call_log 落库（P5）：护栏判定观测写入 MySQL。

P4 判定只走 logger；P5 建 tool_call_log 表落库。观测层对决策层透明：
落库失败仅记 error 日志，绝不阻断决策主流程。
"""
import json
import logging

from app.infrastructure import mysql_pool

logger = logging.getLogger(__name__)

_MAX_QUERY = 500    # query_text 截断
_MAX_SUMMARY = 512  # result_summary 截断


def _result_summary(result: dict | None) -> str:
    """结果摘要（FC 契约信封）：ok=False 且带错误码 → "错误 <code>"；成功按 data 判别。"""
    if not result:
        return ""
    if not result.get("ok"):
        err = result.get("error") or {}
        return f"错误 {err['code']}" if err.get("code") else ""
    data = result.get("data") or {}
    if "results" in data:
        return f"命中 {len(data['results'])} 条"
    if "orders" in data:
        return f"{len(data['orders'])} 条订单"
    if isinstance(data.get("order"), dict):
        o = data["order"]
        return f"订单 {o.get('order_id', '')} 状态 {o.get('status', '')}"
    return json.dumps(data, ensure_ascii=False)[:_MAX_SUMMARY]


async def write_tool_call(*, session_id: str, user_id: int, round_no: int, tool_name: str,
                          args: dict, result: dict | None, latency_ms: int,
                          verdict: str, reason: str, query_text: str) -> None:
    """护栏判定落一条 tool_call_log。失败静默（仅日志），不抛异常影响决策循环。"""
    try:
        await mysql_pool.execute(
            "INSERT INTO tool_call_log (session_id, user_id, round_no, tool_name, args_json, "
            " result_summary, latency_ms, verdict, verdict_reason, query_text) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                session_id[:64], user_id, round_no, tool_name[:64],
                json.dumps(args or {}, ensure_ascii=False), _result_summary(result),
                latency_ms, verdict, (reason or "")[:255], query_text[:_MAX_QUERY],
            ),
        )
    except Exception as exc:
        logger.error("event=tool_call_log_write_error",
                     extra={"tool": tool_name, "verdict": verdict, "error": str(exc)})
