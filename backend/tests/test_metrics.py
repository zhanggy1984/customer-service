"""metrics 轻量 Prometheus 输出测试：格式可被 Prometheus 解析 + 埋点计数正确。"""
import asyncio

from app.infrastructure import metrics


def _reset() -> None:
    """清空全局指标（metrics 为进程内存态，测试间需隔离）。"""
    metrics._counters.clear()
    metrics._sum.clear()
    metrics._count.clear()
    metrics._gauges.clear()


def test_render_empty_is_ok():
    _reset()
    assert metrics.render() == ""


def test_counter_and_summary_output_format():
    _reset()
    metrics.inc("llm_calls", {"model": "deepseek-chat", "ok": "true"})
    metrics.inc("llm_calls", {"model": "deepseek-chat", "ok": "true"})
    metrics.inc("llm_calls", {"model": "deepseek-chat", "ok": "false"})
    metrics.observe("llm_latency_seconds", 0.5, {"model": "deepseek-chat"})
    metrics.observe("llm_latency_seconds", 0.7, {"model": "deepseek-chat"})
    text = metrics.render()
    # counter 带 _total 后缀；标签排序稳定
    assert 'llm_calls_total{model="deepseek-chat",ok="false"} 1' in text
    assert 'llm_calls_total{model="deepseek-chat",ok="true"} 2' in text
    # summary 输出 _sum/_count（可算均值/总量）
    assert 'llm_latency_seconds_sum{model="deepseek-chat"} 1.2' in text
    assert 'llm_latency_seconds_count{model="deepseek-chat"} 2' in text
    assert text.endswith("\n")


def test_gauge_output():
    _reset()
    metrics.set_gauge("queue_depth", 3)
    assert "queue_depth 3" in metrics.render()


def test_intent_rule_hit_increments():
    """intent_rules 命中 → 计数（规则层 vs LLM 层接管比例可量化）"""
    _reset()
    from app.agent import intent_rules

    assert intent_rules.match_intent_rules("你好").intent == "CHITCHAT"
    assert 'intent_rule_hit_total{intent="CHITCHAT"} 1' in metrics.render()


def test_session_lock_timeout_increments(monkeypatch):
    """锁等待超时 → 计数（429 场景可观测）"""
    _reset()
    from app.session import locks as locks_mod

    # 覆盖 conftest 的假 Redis（其 set 立即成功不会超时）：模拟锁一直被占
    class _BusyRedis:
        async def set(self, *a, **kw):
            return None

    monkeypatch.setattr(locks_mod, "_redis", lambda: _BusyRedis())
    monkeypatch.setattr(locks_mod.settings, "session_lock_wait_timeout", 0.05)

    lock = locks_mod.RedisSessionLock("sid-metrics")
    with __import__("pytest").raises(locks_mod.SessionLockTimeoutError):
        asyncio.run(lock._acquire())
    assert "session_lock_timeout_total 1" in metrics.render()
