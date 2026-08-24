"""routes 契约单测：SSE 首帧 meta + greeting 路径补发 usage（契约 §5.1）。

不启动整站（lifespan 连 Redis/MySQL 太重），只挂 routes.router +
override get_current_user + mock session_manager，验证：
- 首帧是 meta（agent/interface/contract_version）
- created_new 会话（greeting 路径，无 LLM）也补发 usage 事件
- 帧格式 event: <type> + data 带 ts（sse_format）
"""
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.api.deps import get_current_user
from app.session.models import Session


def _async_ret(value):
    async def _f(*args, **kwargs):
        return value
    return _f


def _parse_sse(stream: str) -> list[dict]:
    events = []
    for frame in stream.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        lines = frame.split("\n")
        evt = {"event": lines[0].split(":", 1)[1].strip(), "data": None}
        for line in lines[1:]:
            if line.startswith("data:"):
                evt["data"] = json.loads(line[5:].strip())
        events.append(evt)
    return events


def _make_client(monkeypatch) -> TestClient:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = lambda: {"sub": "1", "username": "u"}

    from app.session.manager import session_manager
    created = Session(session_id="new-sess", user_id=1)
    monkeypatch.setattr(session_manager, "get_session", _async_ret(None))  # 会话不存在 → greeting
    monkeypatch.setattr(session_manager, "create_session", _async_ret(created))
    monkeypatch.setattr(session_manager, "update_session", _async_ret(None))
    return TestClient(app)


def test_greeting_path_emits_meta_then_usage_then_done(monkeypatch):
    """created_new 会话：首帧 meta，随后 usage 补发（无 LLM），done 收尾。"""
    client = _make_client(monkeypatch)
    r = client.post("/api/v1/sessions/new-sess/messages", json={"content": "你好"})
    assert r.status_code == 200
    events = _parse_sse(r.text)

    # 首帧 meta（契约 §5.1 可选，声明接口身份）
    assert events[0]["event"] == "meta"
    meta = events[0]["data"]
    assert meta["agent"] == "customer-service"
    assert meta["interface"] == "sessions/{sid}/messages"
    assert meta["contract_version"] == "1.0"
    assert "ts" in meta and isinstance(meta["ts"], int)

    # greeting 路径（created_new 无 LLM）：answer 全量补发（TTFT 起点）+ usage 字段齐全
    types = [e["data"]["type"] for e in events]
    assert "answer" in types
    answer_evt = next(e["data"] for e in events if e["data"]["type"] == "answer")
    assert "您好" in answer_evt["delta"]
    assert "usage" in types
    usage_evt = next(e["data"] for e in events if e["data"]["type"] == "usage")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert field in usage_evt

    # 最后一帧 done
    assert events[-1]["event"] == "done"
    assert "session_id" in events[-1]["data"]
