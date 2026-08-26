"""DeepSeekGateway 流式重试边界单测（mock HTTP，不真实调用）。

覆盖 _stream 的核心决策：
- 首个 content delta 前 HTTP 429/5xx/超时 → 换 Key 重试
- 首个 content delta 后任何异常 → 直接上抛，不回滚已流出 token
- usage chunk（include_usage）→ yield ("", usage, None)，cache 字段透传
"""
import pytest
import httpx

from app.config import settings
from app.infrastructure.deepseek_gateway import (
    DeepSeekGateway,
    LLMUnavailableError,
)


class FakeKey:
    def __init__(self, index: int) -> None:
        self.index = index
        self.api_key = f"test-key-{index}"
        self.rate_limited = False

    async def record_request(self) -> None:
        pass

    async def mark_rate_limited(self, retry_after: int) -> None:
        self.rate_limited = True


class FakePool:
    """select_key 依次弹出 key；all_cooling 恒 False。"""

    def __init__(self, keys: list[FakeKey]) -> None:
        self._keys = list(keys)
        self._idx = 0
        self.select_calls = 0

    async def select_key(self):
        if self._idx >= len(self._keys):
            return None
        k = self._keys[self._idx]
        self._idx += 1
        self.select_calls += 1
        return k

    async def all_cooling(self) -> bool:
        return False


class FakeStreamCtx:
    """模拟 httpx.AsyncClient.stream 的 async context manager。

    responses: [{status, lines, raise_after}]，__aenter__ 弹出一个。
    """

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self._current: dict | None = None
        self.headers: dict = {}  # 429 分支读取 retry-after

    async def __aenter__(self):
        self._current = self._responses.pop(0)
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    @property
    def status_code(self) -> int:
        return self._current["status"]

    async def aiter_lines(self):
        for line in self._current["lines"]:
            yield line
        if self._current.get("raise_after") is not None:
            raise self._current["raise_after"]


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self._ctx = FakeStreamCtx(responses)
        self.stream_kwargs: list[dict] = []  # 捕获每次 stream 调用参数（断言 thinking payload）

    def stream(self, *args, **kwargs):
        self.stream_kwargs.append(kwargs)
        return self._ctx


def _mk_gateway(responses: list[dict], keys: list[FakeKey] | None = None) -> DeepSeekGateway:
    gw = DeepSeekGateway()
    gw._pool = FakePool(keys or [FakeKey(0), FakeKey(1)])
    gw._client = FakeClient(responses)
    return gw


def _d(content: str) -> str:
    return f'data: {content}'


async def _collect(gw: DeepSeekGateway):
    return [item async for item in gw._stream([{"role": "user", "content": "hi"}], "m", None, None)]


@pytest.mark.asyncio
async def test_stream_http_429_before_content_retries():
    """首个 delta 前 HTTP 429 → 冷却换 Key 重试成功。"""
    gw = _mk_gateway([
        {"status": 429, "lines": []},
        {"status": 200, "lines": [_d('{"choices":[{"delta":{"content":"回复"}}]}'), "data: [DONE]"]},
    ], [FakeKey(0), FakeKey(1)])
    items = await _collect(gw)
    assert items == [("回复", None, None)]
    assert gw._pool._keys[0].rate_limited is True
    assert gw._pool._keys[1].rate_limited is False
    assert gw._pool.select_calls == 2


@pytest.mark.asyncio
async def test_stream_http_5xx_before_content_retries():
    """首个 delta 前 5xx → 换 Key 重试成功。"""
    gw = _mk_gateway([
        {"status": 503, "lines": []},
        {"status": 200, "lines": [_d('{"choices":[{"delta":{"content":"ok"}}]}'), "data: [DONE]"]},
    ], [FakeKey(0), FakeKey(1)])
    items = await _collect(gw)
    assert items == [("ok", None, None)]
    assert gw._pool.select_calls == 2


@pytest.mark.asyncio
async def test_stream_timeout_before_content_retries():
    """首个 delta 前流内超时 → 换 Key 重试成功。"""
    gw = _mk_gateway([
        {"status": 200, "lines": [], "raise_after": httpx.ReadTimeout("slow")},
        {"status": 200, "lines": [_d('{"choices":[{"delta":{"content":"ok"}}]}'), "data: [DONE]"]},
    ], [FakeKey(0), FakeKey(1)])
    items = await _collect(gw)
    assert items == [("ok", None, None)]
    assert gw._pool.select_calls == 2


@pytest.mark.asyncio
async def test_stream_no_retry_after_first_delta():
    """首个 delta 后流中断 → 直接上抛 LLMUnavailableError，不重试（不回滚已流出 token）。"""
    gw = _mk_gateway([
        {"status": 200,
         "lines": [_d('{"choices":[{"delta":{"content":"部分"}}]}')],
         "raise_after": httpx.ConnectError("conn reset")},
    ], [FakeKey(0), FakeKey(1)])
    got = []
    with pytest.raises(LLMUnavailableError):
        async for item in gw._stream([{"role": "user", "content": "hi"}], "m", None, None):
            got.append(item)
    assert got == [("部分", None, None)]  # 已流出内容保留
    assert gw._pool.select_calls == 1  # 未换 Key 重试


@pytest.mark.asyncio
async def test_stream_usage_chunk_cache_passthrough():
    """usage chunk 透出 cache hit/miss 字段，且顺序在内容之后。"""
    usage_line = _d('{"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,'
                    '"prompt_cache_hit_tokens":70,"prompt_cache_miss_tokens":30},"choices":[]}')
    gw = _mk_gateway([
        {"status": 200, "lines": [
            _d('{"choices":[{"delta":{"content":"完整"}}]}'),
            usage_line,
            "data: [DONE]",
        ]},
    ], [FakeKey(0)])
    items = await _collect(gw)
    assert items[0] == ("完整", None, None)
    assert items[1] == ("", {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                             "prompt_cache_hit_tokens": 70, "prompt_cache_miss_tokens": 30}, None)


@pytest.mark.asyncio
async def test_stream_retries_exhausted_raises():
    """持续 5xx 重试耗尽 → LLMUnavailableError。"""
    gw = _mk_gateway([
        {"status": 503, "lines": []},
        {"status": 503, "lines": []},
        {"status": 503, "lines": []},
    ], [FakeKey(0), FakeKey(1), FakeKey(2)])
    with pytest.raises(LLMUnavailableError):
        await _collect(gw)
    assert gw._pool.select_calls == 3


@pytest.mark.asyncio
async def test_stream_reasoning_before_content():
    """开启 thinking 时 reasoning_content 增量先于 content 流出（契约第三元素）。"""
    gw = _mk_gateway([
        {"status": 200, "lines": [
            _d('{"choices":[{"delta":{"reasoning_content":"先思考"}}]}'),
            _d('{"choices":[{"delta":{"content":"再回答"}}]}'),
            "data: [DONE]",
        ]},
    ], [FakeKey(0)])
    items = await _collect(gw)
    assert items == [("", None, "先思考"), ("再回答", None, None)]


@pytest.mark.asyncio
async def test_stream_payload_thinking_enabled(monkeypatch):
    """思考开关开启时 payload 携带 thinking.enabled（deepseek-chat 依赖此参数输出 reasoning）。"""
    monkeypatch.setattr(settings, "deepseek_thinking_enabled", True)
    gw = _mk_gateway([
        {"status": 200, "lines": [_d('{"choices":[{"delta":{"content":"ok"}}]}'), "data: [DONE]"]},
    ], [FakeKey(0)])
    await _collect(gw)
    assert gw._client.stream_kwargs[0]["json"]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_stream_payload_thinking_disabled(monkeypatch):
    """思考开关关闭时不带 thinking 参数（回退现状，不传无效参数）。"""
    monkeypatch.setattr(settings, "deepseek_thinking_enabled", False)
    gw = _mk_gateway([
        {"status": 200, "lines": [_d('{"choices":[{"delta":{"content":"ok"}}]}'), "data: [DONE]"]},
    ], [FakeKey(0)])
    await _collect(gw)
    assert "thinking" not in gw._client.stream_kwargs[0]["json"]


@pytest.mark.asyncio
async def test_stream_no_retry_after_reasoning_streamed():
    """reasoning 已流出（started=True）后流中断 → 直接上抛，不重试（重试会重复思考内容）。"""
    gw = _mk_gateway([
        {"status": 200,
         "lines": [_d('{"choices":[{"delta":{"reasoning_content":"思考中"}}]}')],
         "raise_after": httpx.ConnectError("conn reset")},
    ], [FakeKey(0), FakeKey(1)])
    got = []
    with pytest.raises(LLMUnavailableError):
        async for item in gw._stream([{"role": "user", "content": "hi"}], "m", None, None):
            got.append(item)
    assert got == [("", None, "思考中")]  # 已流出思考保留
    assert gw._pool.select_calls == 1  # 未换 Key 重试
