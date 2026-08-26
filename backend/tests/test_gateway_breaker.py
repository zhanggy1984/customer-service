"""LLM 网关熔断 + 重试退避 + 空返回兜底单测（mock HTTP，不真实调用）。

- 熔断：连续 LLMUnavailableError 达阈值 → open → 冷却期入口直接拒绝零网络尝试；半开放行；成功 reset；
  CapacityExceeded/AllKeysDown 不累计（非上游故障）。
- 退避：_call 换 Key 重试按 (0.1, 0.2) 指数退避（参照 DB retry.py 的 BACKOFF_DELAYS 语义）。
- 空返回：chat_stream 只出 usage 无 content → 生成节点回退 _EMPTY_ANSWER_FALLBACK 且未流式
  （finalize 全量补发，保证 token 拼接 == done.content 契约一致）。
"""
import asyncio
import time

import pytest

from app.agent import orchestrator as orch
from app.infrastructure.deepseek_gateway import (
    AllKeysDownError,
    CapacityExceededError,
    DeepSeekGateway,
    LLMUnavailableError,
    StreamInterruptedError,
)


def _mk_gateway() -> DeepSeekGateway:
    return DeepSeekGateway()


# ---------- 熔断 ----------


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold_then_fast_fail(monkeypatch):
    """连续 LLMUnavailableError 达阈值 → 熔断 open → 冷却期 chat() 直接拒绝且零 _call 调用。"""
    gw = _mk_gateway()
    calls = {"n": 0}

    async def fake_call(*a, **kw):
        calls["n"] += 1
        raise LLMUnavailableError("boom")

    monkeypatch.setattr(gw, "_call", fake_call)
    for _ in range(2):  # 前 2 次每次都进 _call（重试耗尽）→ 累计到阈值（挑战2：强故障信号阈值取 2）
        with pytest.raises(LLMUnavailableError):
            await gw.chat([{"role": "user", "content": "hi"}])
    assert gw._breaker["failures"] >= 2
    assert gw._breaker["open_until"] > time.time()

    calls_before = calls["n"]
    with pytest.raises(LLMUnavailableError):
        await gw.chat([{"role": "user", "content": "hi"}])  # 熔断 open → 快速失败
    assert calls["n"] == calls_before  # 零网络尝试（防 LLM 慢挂时每请求 3 连打放大延迟）


@pytest.mark.asyncio
async def test_breaker_half_open_allows_one_after_cooldown(monkeypatch):
    """冷却到期 → 半开放行：failures 重置计数，放行一次真实调用。"""
    gw = _mk_gateway()
    gw._breaker["failures"] = 2
    gw._breaker["open_until"] = time.time() - 1  # 冷却已过

    async def fake_call(*a, **kw):
        return {"ok": True}

    monkeypatch.setattr(gw, "_call", fake_call)
    result = await gw.chat([{"role": "user", "content": "hi"}])
    assert result == {"ok": True}
    assert gw._breaker["failures"] == 0  # 成功后重置


@pytest.mark.asyncio
async def test_breaker_reset_on_success(monkeypatch):
    """失败 1 次后成功一次 → 计数清零（未达阈值不误触发熔断）。"""
    gw = _mk_gateway()
    state = {"fail": 1}

    async def fake_call(*a, **kw):
        if state["fail"] > 0:
            state["fail"] -= 1
            raise LLMUnavailableError("boom")
        return {"ok": True}

    monkeypatch.setattr(gw, "_call", fake_call)
    for _ in range(1):
        with pytest.raises(LLMUnavailableError):
            await gw.chat([{"role": "user", "content": "hi"}])
    assert gw._breaker["failures"] == 1
    await gw.chat([{"role": "user", "content": "hi"}])
    assert gw._breaker["failures"] == 0


@pytest.mark.asyncio
async def test_reset_only_clears_shared_breaker_for_local_broadcaster(monkeypatch):
    """改动3 语义：仅本地广播方 reset 才 close 共享熔断信号，他节点成功不 DEL。

    广播方本地冷却期内被 _breaker_open 拒绝、到不了成功路径，实际触发 close 的
    只有"他节点成功"——其 DEL 会撤销他人广播的共享降级信号，故未广播（open_until=0）
    时 reset 不 close。直接调 _breaker_reset 单测判断逻辑（chat 会因冷却被拒）。
    """
    gw = _mk_gateway()
    close_calls = []

    async def spy_close():
        close_calls.append(1)

    monkeypatch.setattr(gw._cooldown, "close", spy_close)

    # 他节点（本地未广播）：成功 reset 不 close
    await gw._breaker_reset()
    assert close_calls == []

    # 对照：本地广播方（open_until 未来）reset 才 close
    gw._breaker["open_until"] = time.time() + 100
    await gw._breaker_reset()
    assert len(close_calls) == 1


@pytest.mark.asyncio
async def test_breaker_not_counted_for_capacity_or_allkeys_down(monkeypatch):
    """CapacityExceeded / AllKeysDown 不累计熔断（本地排队/429 限流，非上游持续故障）。"""
    gw = _mk_gateway()

    async def fake_capacity(*a, **kw):
        raise CapacityExceededError("busy")

    async def fake_all_keys(*a, **kw):
        raise AllKeysDownError("down")

    monkeypatch.setattr(gw, "_call", fake_capacity)
    with pytest.raises(CapacityExceededError):
        await gw.chat([{"role": "user", "content": "hi"}])
    assert gw._breaker["failures"] == 0

    monkeypatch.setattr(gw, "_call", fake_all_keys)
    with pytest.raises(AllKeysDownError):
        await gw.chat([{"role": "user", "content": "hi"}])
    assert gw._breaker["failures"] == 0


@pytest.mark.asyncio
async def test_breaker_stream_interrupt_not_counted(monkeypatch):
    """流式中断（已产出首个 delta 后异常）→ 冒泡但不累计熔断；未产出的尝试失败才累计。

    挑战1：单次长流中途断（连接抖动/用户断连）是连接级事件，不代表网关整体故障，
    不应把 5 个并发流的自然中断误算成 5 次连续失败把全网关熔断。
    """
    gw = _mk_gateway()

    async def fake_interrupt(*a, **kw):
        if True:
            raise StreamInterruptedError("流式中断")
        yield

    async def fake_fail(*a, **kw):
        if True:
            raise LLMUnavailableError("boom")
        yield

    monkeypatch.setattr(gw, "_stream", fake_interrupt)
    with pytest.raises(StreamInterruptedError):
        async for _ in gw.chat_stream([{"role": "user", "content": "hi"}]):
            pass
    assert gw._breaker["failures"] == 0  # 流中断不累计

    monkeypatch.setattr(gw, "_stream", fake_fail)
    with pytest.raises(LLMUnavailableError):
        async for _ in gw.chat_stream([{"role": "user", "content": "hi"}]):
            pass
    assert gw._breaker["failures"] == 1  # 尝试阶段失败（重试耗尽）才累计


@pytest.mark.asyncio
async def test_breaker_rejects_chat_stream_when_open(monkeypatch):
    """熔断 open 后 chat_stream 同样入口快速拒绝（零 _stream 调用），流式生成不放大慢挂延迟。"""
    gw = _mk_gateway()
    calls = {"n": 0}
    gw._breaker["failures"] = 2
    gw._breaker["open_until"] = time.time() + 60  # 冷却期内

    async def fake_stream(*a, **kw):
        calls["n"] += 1
        yield "x", None

    monkeypatch.setattr(gw, "_stream", fake_stream)
    with pytest.raises(LLMUnavailableError):
        async for _ in gw.chat_stream([{"role": "user", "content": "hi"}]):
            pass
    assert calls["n"] == 0  # 零网络尝试（chat 与 chat_stream 双入口一致拦截）


@pytest.mark.asyncio
async def test_breaker_reset_on_stream_success(monkeypatch):
    """流式调用先失败（未达阈值）再正常流结束 → 计数清零。"""
    gw = _mk_gateway()

    async def fake_fail(*a, **kw):
        if True:
            raise LLMUnavailableError("boom")
        yield

    async def fake_ok(*a, **kw):
        yield "你好", None

    monkeypatch.setattr(gw, "_stream", fake_fail)
    with pytest.raises(LLMUnavailableError):
        async for _ in gw.chat_stream([{"role": "user", "content": "hi"}]):
            pass
    assert gw._breaker["failures"] == 1

    monkeypatch.setattr(gw, "_stream", fake_ok)
    collected = []
    async for item in gw.chat_stream([{"role": "user", "content": "hi"}]):
        collected.append(item)
    assert collected == [("你好", None)]
    assert gw._breaker["failures"] == 0  # 正常流结束 → 成功 reset


def test_llm_fallback_errors_centralized():
    """LLM_FALLBACK_ERRORS 集中定义于 deepseek.py，agent 两侧复用同一元组（修改点4：去重）。

    三类异常必须同时被覆盖：LLMUnavailableError（网络/超时/熔断）、CapacityExceededError
    （本地排队）、AllKeysDownError（429 全冷）——任一遗漏都会让该故障不再走规则引擎兜底。
    """
    from app.agent import agent_loop
    from app.infrastructure.deepseek import LLM_FALLBACK_ERRORS

    assert set(LLM_FALLBACK_ERRORS) == {LLMUnavailableError, CapacityExceededError, AllKeysDownError}
    assert agent_loop.LLM_FALLBACK_ERRORS is LLM_FALLBACK_ERRORS  # 不再本地重复定义
    assert orch.LLM_FALLBACK_ERRORS is LLM_FALLBACK_ERRORS
    # StreamInterruptedError 是 LLMUnavailableError 子类 → except LLM_FALLBACK_ERRORS 自动捕获
    # （熔断/流中断两类都走规则引擎兜底，isinstance 语义，无需显式加入元组）
    assert StreamInterruptedError.__mro__[1] is LLMUnavailableError


# ---------- 退避 ----------


class _FakeKey:
    def __init__(self, index: int) -> None:
        self.index = index
        self.api_key = f"test-key-{index}"
        self.rate_limited = False

    async def record_request(self) -> None:
        pass

    async def get_rpm(self) -> int:
        return 0

    async def mark_rate_limited(self, retry_after: int) -> None:
        self.rate_limited = True


class _FakePool:
    """select_key 依次弹出 key；all_cooling 恒 False。"""

    def __init__(self, keys: list[_FakeKey]) -> None:
        self._keys = list(keys)
        self._idx = 0

    async def select_key(self):
        if self._idx >= len(self._keys):
            return None
        k = self._keys[self._idx]
        self._idx += 1
        return k

    async def all_cooling(self) -> bool:
        return False


class _FakeResp:
    def __init__(self, status: int, data: dict | None = None) -> None:
        self.status_code = status
        self.headers: dict = {}
        self._data = data or {}

    def json(self) -> dict:
        return self._data


@pytest.mark.asyncio
async def test_call_429_retry_with_backoff(monkeypatch):
    """首个 Key 429 → 冷却 + 按退避(0.1)换 Key 重试成功。"""
    gw = _mk_gateway()
    gw._pool = _FakePool([_FakeKey(0), _FakeKey(1)])
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    posts = [_FakeResp(429), _FakeResp(200, {"content": "ok"})]

    async def fake_post(*a, **kw):
        return posts.pop(0)

    gw._client.post = fake_post
    await gw._call([{"role": "user", "content": "hi"}], "m", None)
    assert sleeps == [0.1]  # attempt=0 失败退避 0.1 后换 Key
    assert gw._pool._keys[0].rate_limited is True


@pytest.mark.asyncio
async def test_call_5xx_retry_backoff_escalates_then_unavailable(monkeypatch):
    """连续 5xx → 退避递增 (0.1, 0.2)，耗尽后抛 LLMUnavailableError。"""
    gw = _mk_gateway()
    gw._pool = _FakePool([_FakeKey(0), _FakeKey(1), _FakeKey(2)])
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def fake_post(*a, **kw):
        return _FakeResp(503)

    gw._client.post = fake_post
    with pytest.raises(LLMUnavailableError):
        await gw._call([{"role": "user", "content": "hi"}], "m", None)
    assert sleeps == [0.1, 0.2]  # attempt=0→0.1, attempt=1→0.2


# ---------- 空返回兜底 ----------


def _mk_session(agent_state=None, intent=None):
    from app.session.models import Session

    return Session(session_id="test-s", user_id=1, agent_state=agent_state, intent=intent)


@pytest.mark.asyncio
async def test_compose_policy_answer_empty_content_fallback(monkeypatch):
    """chat_stream 只出 usage 无 content → 兜底话术 + 未流式（finalize 全量补发，契约一致）。"""
    async def fake_stream(messages, model=None, timeout=None, temperature=None):
        yield "", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                   "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}

    monkeypatch.setattr(orch.deepseek_client, "chat_stream", fake_stream)
    reply, streamed = await orch._compose_policy_answer(
        {"search_policy": {"ok": True, "data": {"results": [{"source": "售后政策.md", "text": "七天无理由退货"}]}}},
        "退货政策是什么", None)
    assert reply == orch._EMPTY_ANSWER_FALLBACK
    assert streamed is False  # 未流任何 delta → 不走流式，finalize 补发


@pytest.mark.asyncio
async def test_handle_chitchat_empty_content_fallback(monkeypatch):
    """闲聊 chat_stream 不产出内容 → 兜底话术 + 未流式。"""
    async def fake_stream(messages, model=None, timeout=None, temperature=None):
        if False:
            yield  # async generator：运行时不产出任何内容

    monkeypatch.setattr(orch.deepseek_client, "chat_stream", fake_stream)
    reply, streamed = await orch._handle_chitchat(_mk_session(), "你好", 1)
    assert reply == orch._EMPTY_ANSWER_FALLBACK
    assert streamed is False


@pytest.mark.asyncio
async def test_compose_policy_fallback_answer_empty_body_fallback(monkeypatch):
    """检索故障兜底：LLM 正文空 → 前缀声明 + 兜底话术 + 转人工后缀（不只剩干瘪声明）。"""
    async def fake_stream(messages, model=None, timeout=None, temperature=None):
        if False:
            yield

    monkeypatch.setattr(orch.deepseek_client, "chat_stream", fake_stream)
    reply = await orch._compose_policy_fallback_answer("退货政策是什么")
    assert orch._EMPTY_ANSWER_FALLBACK in reply
    assert "知识库检索暂不可用" in reply  # 前缀低可信度声明保留
    assert "转人工" in reply  # 尾部转人工建议保留


@pytest.mark.asyncio
async def test_compose_policy_answer_whitespace_only_content_contract(monkeypatch):
    """LLM 只 emit 空白 delta（"  " 对 `if delta:` 为 True 会 emit）→ 补发兜底且已流式。

    挑战3：若走"未流式"路径（(fallback, False)），前端已收空白 token.delta + finalize 全量补发
    会拼出"空白+fallback"，与 done.content=fallback 不一致；正确做法是补发兜底进 reply，
    使 token 拼接 == done.content。
    """
    emitted = []

    async def fake_stream(messages, model=None, timeout=None, temperature=None):
        yield "  ", None  # 非空字符串但全空白：生成节点会 emit
        yield "", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                   "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}

    async def emit(ev):
        emitted.append(ev)

    monkeypatch.setattr(orch.deepseek_client, "chat_stream", fake_stream)
    reply, streamed = await orch._compose_policy_answer(
        {"search_policy": {"ok": True, "data": {"results": [{"source": "售后政策.md", "text": "七天无理由退货"}]}}},
        "退货政策是什么", emit)
    assert streamed is True
    assert reply == "  " + orch._EMPTY_ANSWER_FALLBACK
    # 前端 token.delta 序列拼接 == reply == done.content（契约一致）
    assert "".join(ev["delta"] for ev in emitted) == reply


@pytest.mark.asyncio
async def test_handle_chitchat_whitespace_only_content_contract(monkeypatch):
    """闲聊只 emit 空白 delta → 同样补发兜底拼接，契约一致。"""
    emitted = []

    async def fake_stream(messages, model=None, timeout=None, temperature=None):
        yield "  ", None

    async def emit(ev):
        emitted.append(ev)

    monkeypatch.setattr(orch.deepseek_client, "chat_stream", fake_stream)
    reply, streamed = await orch._handle_chitchat(_mk_session(), "你好", 1, emit)
    assert streamed is True
    assert reply == "  " + orch._EMPTY_ANSWER_FALLBACK
    assert "".join(ev["delta"] for ev in emitted) == reply
