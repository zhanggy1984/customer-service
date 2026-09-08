"""obs_sdk llm_call 观测接线单测（§11.3 cs #3：DeepSeekGateway.chat/chat_stream 出口打点）。

直通路径（_obs_sdk() 返 None → 零打点零开销）由既有网关单测天然覆盖（mock HTTP 下业务
语义不变即证明观测不掺入主链路）；本文件只验证**启用观测**分支：
- chat 成功：record_llm ok + usage（非流式响应顶层 usage）透传
- chat 失败：先记 error 再抛（§2.4 前提）；error_type 分型——真实 _call 折叠 5xx 重试耗尽
  经 exc.error_type=HTTP_503 带出（验证 #86 核心改动），熔断类按类映射
- chat_stream 成功：流末 usage chunk → ok 打点，观测不吞/改流事件
- chat_stream 中断：StreamInterruptedError → STREAM_INTERRUPTED；LLMUnavailableError → error_type

全程 monkeypatch _obs_sdk / _call / _stream，不触真实 HTTP、不依赖 sdk 安装、不依赖 Redis
（各用例单次失败熔断计数 < 阈值 2，不触发 cooldown 广播）。
"""
import pytest

from app.config import settings
from app.infrastructure import deepseek_gateway as dgw
from app.infrastructure.deepseek_gateway import (
    AllKeysDownError,
    CapacityExceededError,
    DeepSeekGateway,
    LLMUnavailableError,
    StreamInterruptedError,
)


class _FakeObs:
    """假 obs_sdk：只记录 record_llm 调用（无 init 状态校验）。"""

    def __init__(self):
        self.calls = []

    def record_llm(self, model, status, *, duration_ms, error_type=None,
                   error_msg=None, usage=None):
        self.calls.append({
            "model": model, "status": status, "duration_ms": duration_ms,
            "error_type": error_type, "error_msg": error_msg, "usage": usage,
        })


def _mk_gateway() -> DeepSeekGateway:
    return DeepSeekGateway()


# ---------- chat 出口打点 ----------


@pytest.mark.asyncio
async def test_chat_obs_records_ok_with_usage(monkeypatch):
    """chat 成功 → record_llm ok + 响应顶层 usage 透传（含 cache 字段，消费端只读 3 键）。"""
    fake = _FakeObs()
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: fake)
    gw = _mk_gateway()

    async def fake_call(*a, **kw):
        return {"content": "你好", "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                             "total_tokens": 15,
                                             "prompt_cache_hit_tokens": 7,
                                             "prompt_cache_miss_tokens": 3}}

    monkeypatch.setattr(gw, "_call", fake_call)
    result = await gw.chat([{"role": "user", "content": "hi"}])
    assert result["content"] == "你好", "观测不得吞/改返回值"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["status"] == "ok"
    assert call["usage"]["total_tokens"] == 15
    assert call["usage"]["prompt_cache_hit_tokens"] == 7, "cache 字段原样透传"
    assert call["model"] == settings.deepseek_model_chat
    assert call["duration_ms"] is not None and call["duration_ms"] >= 0
    assert call["error_type"] is None and call["error_msg"] is None


@pytest.mark.asyncio
async def test_chat_obs_error_type_http_5xx_via_real_call(monkeypatch):
    """真实 _call 折叠 5xx 重试耗尽 → LLMUnavailableError(error_type=HTTP_503) → 先记 error 再抛。

    验证 #86 核心链路：_call 在最终 raise 前把最后一次失败类别刻入异常，chat 外层
    record_llm 读 exc.error_type，不再二次猜。
    """
    fake = _FakeObs()
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: fake)
    gw = _mk_gateway()
    gw._pool = _FakePool([_FakeKey(i) for i in range(3)])  # attempt 0/1/2 各弹一 Key

    async def fake_post(*a, **kw):
        return _FakeResp(503)

    gw._client.post = fake_post
    with pytest.raises(LLMUnavailableError):
        await gw.chat([{"role": "user", "content": "hi"}])
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["status"] == "error"
    assert call["error_type"] == "HTTP_503", "last_fail_type 折叠 5xx → HTTP_503 带出"
    assert "LLM 调用失败" in call["error_msg"]
    assert call["usage"] is None


@pytest.mark.asyncio
async def test_chat_obs_error_type_breaker_classes(monkeypatch):
    """_call 抛熔断类异常（无 error_type 字段）→ 按类映射 ALL_KEYS_DOWN / QUEUE_TIMEOUT。"""
    fake = _FakeObs()
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: fake)
    gw = _mk_gateway()

    async def fake_all_keys(*a, **kw):
        raise AllKeysDownError("所有 DeepSeek Key 均不可用")

    monkeypatch.setattr(gw, "_call", fake_all_keys)
    with pytest.raises(AllKeysDownError):
        await gw.chat([{"role": "user", "content": "hi"}])
    assert fake.calls[0]["status"] == "error"
    assert fake.calls[0]["error_type"] == "ALL_KEYS_DOWN"

    async def fake_capacity(*a, **kw):
        raise CapacityExceededError("系统繁忙")

    monkeypatch.setattr(gw, "_call", fake_capacity)
    with pytest.raises(CapacityExceededError):
        await gw.chat([{"role": "user", "content": "hi"}])
    assert fake.calls[1]["status"] == "error"
    assert fake.calls[1]["error_type"] == "QUEUE_TIMEOUT"


# ---------- chat_stream 出口打点 ----------


@pytest.mark.asyncio
async def test_chat_stream_obs_records_ok_with_stream_usage(monkeypatch):
    """chat_stream 正常流末 → record_llm ok + 流末 usage chunk 透传，观测不吞/改流事件。"""
    fake = _FakeObs()
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: fake)
    gw = _mk_gateway()

    async def fake_stream(*a, **kw):
        yield "你好", None, None
        yield "", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, None

    monkeypatch.setattr(gw, "_stream", fake_stream)
    out = [item async for item in gw.chat_stream([{"role": "user", "content": "hi"}])]
    assert [i[0] for i in out] == ["你好", ""], "观测不得吞/改流事件"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["status"] == "ok"
    assert call["usage"]["total_tokens"] == 15, "usage 应从流末 chunk 透传"
    assert call["error_type"] is None and call["error_msg"] is None


@pytest.mark.asyncio
async def test_chat_stream_obs_interrupt_records_then_raises(monkeypatch):
    """流中断（已产出首个 delta）→ 先记 error+STREAM_INTERRUPTED 再抛（§2.4 前提）。"""
    fake = _FakeObs()
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: fake)
    gw = _mk_gateway()

    async def fake_stream(*a, **kw):
        yield "部分", None, None
        raise StreamInterruptedError("LLM 流式中断")

    monkeypatch.setattr(gw, "_stream", fake_stream)
    got = []
    with pytest.raises(StreamInterruptedError):
        async for item in gw.chat_stream([{"role": "user", "content": "hi"}]):
            got.append(item)
    assert [i[0] for i in got] == ["部分"], "已流出内容保留、先记再抛"
    assert fake.calls[0]["status"] == "error"
    assert fake.calls[0]["error_type"] == "STREAM_INTERRUPTED"
    assert fake.calls[0]["error_msg"] == "LLM 流式中断"


@pytest.mark.asyncio
async def test_chat_stream_obs_error_type_from_unavailable(monkeypatch):
    """_stream 重试耗尽折叠 → LLMUnavailableError(error_type=HTTP_503) → error_type 透传。"""
    fake = _FakeObs()
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: fake)
    gw = _mk_gateway()

    async def fake_stream(*a, **kw):
        if True:  # async generator（含 yield 语句）；调用即抛、不产出
            raise LLMUnavailableError("LLM 调用失败，请稍后重试", error_type="HTTP_503")
        yield

    monkeypatch.setattr(gw, "_stream", fake_stream)
    with pytest.raises(LLMUnavailableError):
        async for _ in gw.chat_stream([{"role": "user", "content": "hi"}]):
            pass
    assert fake.calls[0]["status"] == "error"
    assert fake.calls[0]["error_type"] == "HTTP_503"


# ---------- 直通路径（观测未启用）不该有打点 ----------


@pytest.mark.asyncio
async def test_chat_no_obs_no_record(monkeypatch):
    """_obs_sdk() 未 init（返 None）→ chat 照常成功、零打点（默认环境即此路径）。"""
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: None)
    gw = _mk_gateway()

    async def fake_call(*a, **kw):
        return {"content": "ok"}

    monkeypatch.setattr(gw, "_call", fake_call)
    result = await gw.chat([{"role": "user", "content": "hi"}])
    assert result["content"] == "ok"


# ---------- 底层 seam（仿 test_gateway_breaker.py 的 _FakePool/_FakeKey/_FakeResp） ----------


class _FakeKey:
    def __init__(self, index: int) -> None:
        self.index = index
        self.api_key = f"mock-key-{index}"  # 值含 mock：全局 pre-commit 对 api_key= 测试桩的豁免约定
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
