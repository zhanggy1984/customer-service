"""usage 聚合器（contextvar）：一轮对话内多次 LLM 调用的 token 全计。

run_agent 入口 begin() 重置，各 LLM 调用点 accumulate()，done 前 emit 单条 usage。
用 contextvar 而非参数传递：intent/complaint_flow 等调用点无需改函数签名。
contextvar 按 asyncio task 隔离，多 session 并发不串。
"""
from contextvars import ContextVar

_ctx: ContextVar[dict | None] = ContextVar("usage_agg", default=None)


def begin() -> None:
    """本轮对话开始，重置聚合器。"""
    _ctx.set({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
              "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0})


def accumulate(usage: dict | None) -> None:
    """累加一次 LLM 调用的 usage。None（流式失败/规则兜底）安全跳过。"""
    agg = _ctx.get()
    if agg is None or not usage:
        return
    agg["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
    agg["completion_tokens"] += usage.get("completion_tokens", 0) or 0
    agg["total_tokens"] += usage.get("total_tokens", 0) or 0
    # 7.4 cache 字段分别累加（命中/未命中各自累计，透传评测平台）
    agg["prompt_cache_hit_tokens"] += usage.get("prompt_cache_hit_tokens", 0) or 0
    agg["prompt_cache_miss_tokens"] += usage.get("prompt_cache_miss_tokens", 0) or 0


def current() -> dict:
    """当前聚合值。未 begin（异常路径）时返回全 0，保证 usage 事件字段齐全。"""
    agg = _ctx.get()
    if agg is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}
    return dict(agg)
