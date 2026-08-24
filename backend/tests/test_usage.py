"""usage 聚合器单测：cache 字段累加（7.4c 透传 cache hit/miss 到评测平台）。

覆盖 begin/accumulate/current 三段：cache 字段与 prompt/completion 同步累加，
未 begin（异常路径）current 返回全 0 保证 usage 事件字段齐全。
"""
from app.agent import usage as usage_mod


def _reset() -> None:
    """重置 contextvar（测试间隔离；否则 begin/current 状态跨测试残留）。"""
    usage_mod._ctx.set(None)


def test_begin_current_all_zero() -> None:
    """begin 后 current 全 0，且字段齐全（含 cache hit/miss）。"""
    _reset()
    usage_mod.begin()
    cur = usage_mod.current()
    assert cur == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                   "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}


def test_accumulate_cache_tokens() -> None:
    """多次 accumulate 分别累加 cache hit/miss 字段。"""
    _reset()
    usage_mod.begin()
    usage_mod.accumulate({"prompt_tokens": 100, "completion_tokens": 50,
                          "total_tokens": 150, "prompt_cache_hit_tokens": 60,
                          "prompt_cache_miss_tokens": 40})
    usage_mod.accumulate({"prompt_tokens": 200, "completion_tokens": 80,
                          "total_tokens": 280, "prompt_cache_hit_tokens": 120,
                          "prompt_cache_miss_tokens": 80})
    cur = usage_mod.current()
    assert cur["prompt_tokens"] == 300
    assert cur["completion_tokens"] == 130
    assert cur["total_tokens"] == 430
    assert cur["prompt_cache_hit_tokens"] == 180
    assert cur["prompt_cache_miss_tokens"] == 120


def test_accumulate_none_skipped() -> None:
    """None（流式失败/规则兜底）安全跳过，不改变聚合值。"""
    _reset()
    usage_mod.begin()
    usage_mod.accumulate({"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15, "prompt_cache_hit_tokens": 6,
                          "prompt_cache_miss_tokens": 4})
    usage_mod.accumulate(None)
    cur = usage_mod.current()
    assert cur["prompt_tokens"] == 10
    assert cur["prompt_cache_hit_tokens"] == 6
    assert cur["prompt_cache_miss_tokens"] == 4


def test_current_without_begin_all_zero() -> None:
    """未 begin（异常路径）current 返回全 0，usage 事件字段齐全。"""
    _reset()
    cur = usage_mod.current()
    assert cur == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                   "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}
