"""LLM 硬失败 → request 终态记 error 的单测（观测语义修正，非业务语义变更）。

背景：SSE 接口恒返回 HTTP 200，观测中间件原判据只有「断连 / HTTP≥400 / ok」，
「LLM 挂了、用户拿到兜底话术」被记成 status=ok。平台判定侧（online classify.py 的
兜底吸收门只认 root 终态 root_status）据此把真实故障当作「已被业务吸收」切掉 ⇒
真实链路产不出回流候选。本文件覆盖三处接线：

1. mark_llm_hard_fail 语义（无上下文 no-op / 首次优先）
2. deepseek_gateway 三处失败出口各自置位（LLMUnavailable / AllKeysDown+Capacity / Cancelled）
3. _obs_end 四分支优先级（断连 > LLM 硬失败 > HTTP 状态码 > ok）

**反向对照**：三处都各有一条「未触发时不得置位/不得记 error」的断言 —— 否则「全记 error」
也能让正向断言通过。
"""
import asyncio
from types import SimpleNamespace

import pytest

from app.infrastructure import deepseek_gateway as dgw
from app.infrastructure.deepseek_gateway import (
    AllKeysDownError,
    CapacityExceededError,
    DeepSeekGateway,
    LLMUnavailableError,
)
from app.main import _obs_end
from app.utils.trace import llm_health_var, mark_llm_hard_fail


class _FakeObs:
    """假 obs_sdk：记录 record_llm / end_request 调用。"""

    def __init__(self):
        self.llm_calls = []
        self.end_calls = []

    def record_llm(self, model, status, *, duration_ms, error_type=None,
                   error_msg=None, usage=None):
        self.llm_calls.append({"status": status, "error_type": error_type})

    def end_request(self, status, *, error_type=None, error_msg=None, input=None, **kw):
        self.end_calls.append({"status": status, "error_type": error_type, "input": input})


def _marker() -> dict:
    m = {"hard_fail": False, "error_type": None}
    llm_health_var.set(m)
    return m


# ---------- 1. 标记语义 ----------


def test_mark_without_request_context_is_noop():
    """无请求上下文（后台任务等）不报错、不产生全局状态。"""
    llm_health_var.set(None)
    mark_llm_hard_fail("llm_timeout")  # 不抛即过


def test_mark_first_failure_wins():
    """一轮内多次硬失败只取首次（先发生的词更具代表性）。"""
    m = _marker()
    mark_llm_hard_fail("llm_timeout")
    mark_llm_hard_fail("llm_other")
    assert m == {"hard_fail": True, "error_type": "llm_timeout"}


# ---------- 2. 网关三处失败出口置位 ----------


def _mk_gateway(monkeypatch) -> DeepSeekGateway:
    """构造网关并摘掉 Redis 依赖（熔断计数/指标不参与本组断言）。"""
    gw = DeepSeekGateway()
    monkeypatch.setattr(gw, "_record_call", lambda *a, **kw: None)

    async def _noop():
        return None

    monkeypatch.setattr(gw, "_breaker_fail", _noop)
    monkeypatch.setattr(gw, "_breaker_reset", _noop)
    monkeypatch.setattr(gw, "_breaker_open", lambda: _false_awaitable())
    return gw


async def _false_awaitable() -> bool:
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize("exc,expected", [
    (LLMUnavailableError("网络失败"), "llm_other"),
    (AllKeysDownError("全部 Key 不可用"), "llm_other"),
    (CapacityExceededError("系统繁忙"), "llm_other"),
])
async def test_gateway_failure_paths_mark(monkeypatch, exc, expected):
    """三类失败出口都要置位（漏一处＝该类故障在后端仍然记 ok）。"""
    m = _marker()
    gw = _mk_gateway(monkeypatch)
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: _FakeObs())

    async def boom(*a, **kw):
        raise exc

    monkeypatch.setattr(gw, "_call", boom)
    with pytest.raises(type(exc)):
        await gw.chat([{"role": "user", "content": "hi"}])
    assert m["hard_fail"] is True
    assert m["error_type"] == expected


@pytest.mark.asyncio
async def test_gateway_cancel_marks(monkeypatch):
    """取消窗口（客户断连）：CancelledError 承 BaseException，同样要置位。"""
    m = _marker()
    gw = _mk_gateway(monkeypatch)
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: _FakeObs())

    async def cancel(*a, **kw):
        raise asyncio.CancelledError()

    monkeypatch.setattr(gw, "_call", cancel)
    with pytest.raises(asyncio.CancelledError):
        await gw.chat([{"role": "user", "content": "hi"}])
    assert m["hard_fail"] is True


@pytest.mark.asyncio
async def test_gateway_success_does_not_mark(monkeypatch):
    """反向对照：调用成功不得置位（否则所有请求都会被记 error）。"""
    m = _marker()
    gw = _mk_gateway(monkeypatch)
    monkeypatch.setattr(dgw, "_obs_sdk", lambda: _FakeObs())

    async def ok_call(*a, **kw):
        return {"content": "ok", "usage": {"total_tokens": 1}}

    monkeypatch.setattr(gw, "_call", ok_call)
    await gw.chat([{"role": "user", "content": "hi"}])
    assert m["hard_fail"] is False
    assert m["error_type"] is None


# ---------- 3. 出口四分支优先级 ----------


def _req(**kw):
    return SimpleNamespace(state=SimpleNamespace(**kw))


def _resp(code=200):
    return SimpleNamespace(status_code=code)


def test_obs_end_hard_fail_beats_http_200():
    """核心判据：LLM 硬失败 + HTTP 200（SSE 恒真）→ 必须记 error 且带白名单词。"""
    obs = _FakeObs()
    _obs_end(obs, _resp(200), _req(),
             {"hard_fail": True, "error_type": "llm_timeout"})
    assert obs.end_calls == [{"status": "error", "error_type": "llm_timeout", "input": None}]


def test_obs_end_reverse_control_stays_ok():
    """反向对照：未发生 LLM 硬失败 + 200 → 必须仍是 ok。"""
    obs = _FakeObs()
    _obs_end(obs, _resp(200), _req(),
             {"hard_fail": False, "error_type": None})
    assert obs.end_calls == [{"status": "ok", "error_type": None, "input": None}]


def test_obs_end_disconnect_beats_hard_fail():
    """断连优先于 LLM 硬失败（客户端没拿到响应是更靠前的事实）。"""
    obs = _FakeObs()
    _obs_end(obs, _resp(200), _req(obs_aborted=True),
             {"hard_fail": True, "error_type": "llm_timeout"})
    assert obs.end_calls[0]["error_type"] == "CLIENT_DISCONNECT"


def test_obs_end_hard_fail_beats_http_5xx():
    """LLM 硬失败优先于状态码：根因是 LLM，不是 HTTP 层。"""
    obs = _FakeObs()
    _obs_end(obs, _resp(500), _req(),
             {"hard_fail": True, "error_type": "llm_connection"})
    assert obs.end_calls == [{"status": "error", "error_type": "llm_connection", "input": None}]


def test_obs_end_http_error_kept_when_no_hard_fail():
    """反向对照：无 LLM 硬失败时 HTTP≥400 仍按 HTTP_xxx 记（不吞既有行为）。"""
    obs = _FakeObs()
    _obs_end(obs, _resp(500), _req(),
             {"hard_fail": False, "error_type": None})
    assert obs.end_calls == [{"status": "error", "error_type": "HTTP_500", "input": None}]


def test_obs_end_carries_input():
    """入参随出口转发（失败 trace 建簇依赖它）。"""
    obs = _FakeObs()
    _obs_end(obs, _resp(200), _req(obs_input={"content": "你好"}),
             {"hard_fail": True, "error_type": "llm_timeout"})
    assert obs.end_calls[0]["input"] == {"content": "你好"}
