"""SSE 帧格式与事件契约单测（评测契约 §5.1）。

覆盖 sse_format 新帧格式（event: 行 + data 内嵌 ts）与 usage/done/answer 等
事件构造函数的字段契约，防止帧格式变更再次破坏前端 useSSE.ts。
"""
import json

from app.agent.response import (
    answer_event,
    done_event,
    error_event,
    meta_event,
    reasoning_event,
    sse_format,
    status_event,
    tool_call_event,
    usage_event,
)


def test_sse_format_frame_lines():
    """新帧格式：首行 event: <type>，次行 data: <json>，帧尾空行。"""
    frame = sse_format({"type": "answer", "delta": "你好"})
    lines = frame.split("\n")
    assert lines[0] == "event: answer"
    assert lines[1].startswith("data: ")
    assert frame.endswith("\n\n")
    payload = json.loads(lines[1][6:])  # 去掉 "data: "
    assert payload["delta"] == "你好"
    assert "ts" in payload and isinstance(payload["ts"], int)


def test_sse_format_ensure_ascii_false():
    """中文不转义为 \\uXXXX，前端可读。"""
    frame = sse_format(answer_event("中文回复"))
    assert "中文回复" in frame
    assert "\\u" not in frame


def test_answer_event_delta_field():
    """answer 事件 delta 增量字段（评测端首个 answer.delta 即 TTFT 起点）。"""
    assert answer_event("增量") == {"type": "answer", "delta": "增量"}


def test_usage_event_fields_complete():
    """usage 事件必选且字段齐全（含 cache hit/miss 透传）。"""
    u = usage_event({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
                     "prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 5})
    assert u["type"] == "usage"
    for field in ("prompt_tokens", "completion_tokens", "total_tokens",
                  "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        assert field in u


def test_done_event_intent_field():
    """done 事件携带 intent，结束一轮。"""
    assert done_event("REFUND_REQUEST") == {"type": "done", "intent": "REFUND_REQUEST"}


def test_aux_event_types():
    """tool_call/status/reasoning/meta/error 事件结构。"""
    assert tool_call_event({"name": "query_order", "args": {}}) == {
        "type": "tool_call", "name": "query_order", "args": {}}
    assert status_event("verify", "校验中") == {"type": "status", "stage": "verify", "message": "校验中"}
    assert reasoning_event("思考中") == {"type": "reasoning", "delta": "思考中"}
    assert meta_event({"contract_version": "1.0"}) == {"type": "meta", "contract_version": "1.0"}
    assert error_event("系统异常") == {"type": "error", "message": "系统异常"}
