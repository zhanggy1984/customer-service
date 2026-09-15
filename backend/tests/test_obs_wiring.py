"""obs_sdk llm_call 观测接线单测（§11.3 cs #3：DeepSeekGateway.chat/chat_stream 出口打点）。

直通路径（_obs_sdk() 返 None → 零打点零开销）由既有网关单测天然覆盖（mock HTTP 下业务
语义不变即证明观测不掺入主链路）；本文件只验证**启用观测**分支：
- chat 成功：record_llm ok + usage（非流式响应顶层 usage）透传
- chat 失败：先记 error 再抛（§2.4 前提）；error_type 分型——真实 _call 折叠 5xx 重试耗尽
  经 exc.error_type=HTTP_503 带出（验证 #86 核心改动），熔断类按类映射，**上报前折叠进平台白名单**
- chat_stream 成功：流末 usage chunk → ok 打点，观测不吞/改流事件
- chat_stream 中断：StreamInterruptedError → llm_connection；LLMUnavailableError → error_type 折叠进白名单

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
    """真实 _call 折叠 5xx 重试耗尽 → LLMUnavailableError(error_type=HTTP_503) → 先记 error 再抛（折叠为 llm_other）。

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
    assert call["error_type"] == "llm_other", "5xx 无对应白名单词，带出后折叠为 llm_other"
    assert "LLM 调用失败" in call["error_msg"]
    assert call["usage"] is None


@pytest.mark.asyncio
async def test_chat_obs_error_type_breaker_classes(monkeypatch):
    """_call 抛熔断类异常（无 error_type 字段）→ 按类映射后仍折叠进白名单（均 llm_other）。"""
    fake = _FakeObs()
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: fake)
    gw = _mk_gateway()

    async def fake_all_keys(*a, **kw):
        raise AllKeysDownError("所有 DeepSeek Key 均不可用")

    monkeypatch.setattr(gw, "_call", fake_all_keys)
    with pytest.raises(AllKeysDownError):
        await gw.chat([{"role": "user", "content": "hi"}])
    assert fake.calls[0]["status"] == "error"
    assert fake.calls[0]["error_type"] == "llm_other"

    async def fake_capacity(*a, **kw):
        raise CapacityExceededError("系统繁忙")

    monkeypatch.setattr(gw, "_call", fake_capacity)
    with pytest.raises(CapacityExceededError):
        await gw.chat([{"role": "user", "content": "hi"}])
    assert fake.calls[1]["status"] == "error"
    assert fake.calls[1]["error_type"] == "llm_other"


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
    """流中断（已产出首个 delta）→ 先记 error（折叠 llm_connection）再抛（§2.4 前提）。"""
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
    assert fake.calls[0]["error_type"] == "llm_connection"
    assert fake.calls[0]["error_msg"] == "LLM 流式中断"


@pytest.mark.asyncio
async def test_chat_stream_obs_error_type_from_unavailable(monkeypatch):
    """_stream 重试耗尽折叠 → LLMUnavailableError(error_type=HTTP_503) → 带出后折叠为白名单词。"""
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
    assert fake.calls[0]["error_type"] == "llm_other"


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


# 平台错误分类白名单（§4.3 L1+L2 词表）。llm_call 的 error_type 若落在此集合外，
# 平台不产生回流候选 ⇒ 值域卫生是本表唯一护栏。
_PLATFORM_ERR_WHITELIST = {
    "llm_timeout", "llm_rate_limit", "llm_connection", "llm_context_exceeded",
    "llm_empty_response", "llm_parse_error", "llm_other",
    "llm_interface_business", "external_non_llm", "db_error", "redis_error",
}


@pytest.mark.parametrize("exc,expected", [
    (LLMUnavailableError("x", error_type="HTTP_429"), "llm_rate_limit"),
    (LLMUnavailableError("x", error_type="HTTP_503"), "llm_other"),
    (LLMUnavailableError("x", error_type="HTTP_401"), "llm_other"),
    (LLMUnavailableError("x", error_type="TIMEOUT"), "llm_timeout"),
    (LLMUnavailableError("x", error_type="NETWORK"), "llm_connection"),
    (LLMUnavailableError("x"), "llm_other"),
    (AllKeysDownError("全冷却"), "llm_other"),
    (CapacityExceededError("排队超时"), "llm_other"),
    (StreamInterruptedError("流中断"), "llm_connection"),
])
def test_llm_error_type_maps_into_platform_whitelist(exc, expected):
    """每个分支的返回值都必须在白名单内（网关内部细词 HTTP_{code}/TIMEOUT/NETWORK 均不在册）。"""
    got = dgw._llm_error_type(exc, "LLM_ERROR")
    assert got == expected
    assert got in _PLATFORM_ERR_WHITELIST, f"{got} 不在平台白名单，平台不会据此产生回流候选"
