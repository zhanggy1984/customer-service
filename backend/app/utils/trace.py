"""链路追踪：请求级 traceId 注入 JSON 日志（T9 网关接入）。

网关 api-gateway 生成 X-Request-ID 头透传至此；若直连后端（不经网关），
中间件自动生成 uuid 兜底。TraceIdFilter 把 trace_id 写入 LogRecord，JsonFormatter
会自动合并非保留字段，因此 formatter 零改动即可输出 trace_id。
"""
import contextvars
import logging

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


class TraceIdFilter(logging.Filter):
    """把当前请求的 trace_id 注入日志记录（JsonFormatter 自动输出该字段）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True
