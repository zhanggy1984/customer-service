"""链路追踪：请求级 traceId 注入 JSON 日志（T9 网关接入）。

网关 api-gateway 生成 X-Request-ID 头透传至此；若直连后端（不经网关），
中间件自动生成 uuid 兜底。TraceIdFilter 把 trace_id 写入 LogRecord，JsonFormatter
会自动合并非保留字段，因此 formatter 零改动即可输出 trace_id。
"""
import contextvars
import logging

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")

# 请求级 LLM 健康标记（观测出口判 request 终态用）。
# 为什么需要：SSE 接口恒返回 HTTP 200，观测中间件若只看状态码，会把「LLM 挂了、用户拿到
# 兜底话术」记成 status=ok。平台判定侧（online classify.py 的兜底吸收门只认 root_status）
# 据此把这个 trace 当作「子节点错误已被业务吸收」切掉 —— 真实故障因此产不出回流候选。
# 怎么做：中间件在建请求上下文时置一个可变 dict，下游失败出口就地改它（同一对象，
# 跨任务可见，无需把 request 传进基础设施层）。dict 而非标量：contextvar 的写发生在
# 子任务上下文里，标量赋值传不回中间件。
llm_health_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "llm_health", default=None
)


def mark_llm_hard_fail(error_type: str) -> None:
    """记「本轮最终未拿到 LLM 结果」（KeyPool 重试已耗尽，非单次失败）。

    一轮内多次硬失败只取首次（先发生的那个词更具代表性）。无请求上下文（后台任务等）
    则忽略——标记只服务于 request 出口。
    """
    marker = llm_health_var.get()
    if marker is not None and not marker["hard_fail"]:
        marker["hard_fail"] = True
        marker["error_type"] = error_type


class TraceIdFilter(logging.Filter):
    """把当前请求的 trace_id 注入日志记录（JsonFormatter 自动输出该字段）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True
