"""tool_call_log 落库（P5）：护栏判定观测写入 MySQL。

P4 判定只走 logger；P5 建 tool_call_log 表落库。观测层对决策层透明：
落库失败仅记 error 日志，绝不阻断决策主流程。
"""
import json
import logging

from app.infrastructure.mysql import mysql_pool

logger = logging.getLogger(__name__)

_MAX_QUERY = 500    # query_text 截断
_MAX_SUMMARY = 512  # result_summary 截断


def _result_summary(result: dict | None) -> str:
    """结果摘要：命中数优先（results/orders 列表），兜底 JSON 截断。"""
    if not result:
        return ""
    if "results" in result:
        return f"命中 {len(result['results'])} 条"
    if "orders" in result:
        return f"{len(result['orders'])} 条订单"
    if isinstance(result.get("order"), dict):
        o = result["order"]
        return f"订单 {o.get('order_id', '')} 状态 {o.get('status', '')}"
    return json.dumps(result, ensure_ascii=False)[:_MAX_SUMMARY]


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
