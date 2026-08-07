"""会话历史接口单元测试：GET /sessions（列表）、GET /sessions/{sid}/messages（历史）、DELETE /sessions/{sid}。

不依赖真实服务：monkeypatch 替换 routes 模块的 mysql_pool / session_manager。
"""
import json
from datetime import datetime, timezone

import pytest

from app.api import routes
from app.session.models import Message, Session


class FakeMySQL:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.sqls: list[str] = []
        self.params: list = []

    async def fetchall(self, sql, params=None):
        self.sqls.append(sql)
        self.params.append(params)
        return self.rows


class FakeSessionManager:
    def __init__(self, session=None):
        self.session = session
        self.closed: list[str] = []

    async def get_session(self, sid):
        return self.session

    async def close_session(self, sid):
        self.closed.append(sid)


def _row(session_id="s1", intent="RETURN_REQUEST", messages=None, created_at=None):
    return {
        "session_id": session_id,
        "intent": intent,
        "messages": json.dumps(messages or [], ensure_ascii=False),
        "created_at": created_at or datetime(2026, 8, 7, 12, 30, 0),
    }


# ---------- GET /sessions ----------

@pytest.mark.asyncio
async def test_list_sessions_filters_user_and_empty(monkeypatch):
    """列表 SQL 只查当前用户、过滤空会话；标题取首条 user 消息（Python 端处理）。"""
    # 空会话过滤发生在 SQL WHERE（messages<>'[]'），mock 层不模拟 SQL，故此处只给有消息的行
    fake = FakeMySQL([
        _row("s1", messages=[{"role": "user", "content": "我要退货"}], created_at=datetime(2026, 8, 7, 12, 0)),
        _row("s2", messages=[{"role": "assistant", "content": "您好"}], created_at=datetime(2026, 8, 7, 13, 0)),
    ])
    monkeypatch.setattr(routes, "mysql_pool", fake)

    resp = await routes.list_sessions(user={"sub": "2"})

    # WHERE 携带当前用户 + 空会话过滤（JSON_LENGTH>0）+ LIMIT 50
    assert "user_id=%s" in fake.sqls[0]
    assert "JSON_LENGTH(messages) > 0" in fake.sqls[0]
    assert "LIMIT 50" in fake.sqls[0]
    assert fake.params[0][0] == 2
    # s1 标题取首条 user 消息；assistant 开头的会话标题为"新会话"
    assert [i["session_id"] for i in resp["items"]] == ["s1", "s2"]
    assert resp["items"][0]["title"] == "我要退货"
    assert resp["items"][1]["title"] == "新会话"
    assert resp["total"] == 2


@pytest.mark.asyncio
async def test_list_sessions_title_truncated_to_30_chars(monkeypatch):
    """首条 user 消息超过 30 字时截断。"""
    long_msg = "长" * 40
    fake = FakeMySQL([_row(messages=[{"role": "user", "content": long_msg}])])
    monkeypatch.setattr(routes, "mysql_pool", fake)

    resp = await routes.list_sessions(user={"sub": "2"})

    assert len(resp["items"][0]["title"]) == 30


@pytest.mark.asyncio
async def test_list_sessions_created_at_iso(monkeypatch):
    """datetime 字段序列化为 ISO 字符串（FastAPI 可 JSON 化），字段名为 updated_at（语义=最后保存时间）。"""
    fake = FakeMySQL([_row(messages=[{"role": "user", "content": "hi"}])])
    monkeypatch.setattr(routes, "mysql_pool", fake)

    resp = await routes.list_sessions(user={"sub": "2"})

    assert resp["items"][0]["updated_at"] == "2026-08-07T12:30:00"


@pytest.mark.asyncio
async def test_list_sessions_dirty_messages_not_list(monkeypatch):
    """messages 列脏数据（合法 JSON 但非数组）不抛异常，标题兜底为"新会话"。"""
    fake = FakeMySQL([
        {"session_id": "s1", "intent": None, "messages": '"not-a-list"', "created_at": datetime(2026, 8, 7, 12, 0)},
    ])
    monkeypatch.setattr(routes, "mysql_pool", fake)

    resp = await routes.list_sessions(user={"sub": "2"})

    assert resp["items"][0]["title"] == "新会话"
    assert resp["total"] == 1


@pytest.mark.asyncio
async def test_list_sessions_dirty_array_elements(monkeypatch):
    """messages 数组内混入非 dict / content 非字符串元素时不抛异常，跳过并取下一个合法 user 消息。"""
    fake = FakeMySQL([
        _row(messages=["hello", 123, {"role": "user", "content": 99}, {"role": "user", "content": "正常消息"}]),
    ])
    monkeypatch.setattr(routes, "mysql_pool", fake)

    resp = await routes.list_sessions(user={"sub": "2"})

    assert resp["items"][0]["title"] == "正常消息"
    assert resp["total"] == 1


# ---------- GET /sessions/{sid}/messages ----------

@pytest.mark.asyncio
async def test_get_messages_ok(monkeypatch):
    """正常返回历史消息，ts 转 ISO 字符串；只返回最近 200 条以内。"""
    ts = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    session = Session(
        session_id="s1",
        user_id=2,
        intent="RETURN_REQUEST",
        messages=[Message(role="user", content="我要退货", ts=ts)],
    )
    monkeypatch.setattr(routes, "session_manager", FakeSessionManager(session))

    resp = await routes.get_session_messages("s1", user={"sub": "2"})

    assert resp["session_id"] == "s1"
    assert resp["intent"] == "RETURN_REQUEST"
    assert resp["messages"][0]["role"] == "user"
    assert resp["messages"][0]["content"] == "我要退货"
    # pydantic model_dump(mode="json") 对 UTC 时间输出 Z 后缀（ISO 8601）
    assert resp["messages"][0]["ts"] == "2026-08-07T12:00:00Z"


@pytest.mark.asyncio
async def test_get_messages_not_found(monkeypatch):
    """会话不存在（Redis/MySQL 均无）→ 404。"""
    monkeypatch.setattr(routes, "session_manager", FakeSessionManager(None))

    with pytest.raises(Exception) as exc_info:
        await routes.get_session_messages("ghost", user={"sub": "2"})
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_messages_forbidden(monkeypatch):
    """会话归属他人 → 403。"""
    session = Session(session_id="s1", user_id=3)
    monkeypatch.setattr(routes, "session_manager", FakeSessionManager(session))

    with pytest.raises(Exception) as exc_info:
        await routes.get_session_messages("s1", user={"sub": "2"})
    assert exc_info.value.status_code == 403


# ---------- DELETE /sessions/{sid} ----------

@pytest.mark.asyncio
async def test_delete_session_ok(monkeypatch):
    """归属校验通过 → close_session 被调用（Redis + MySQL 一并清除）。"""
    session = Session(session_id="s1", user_id=2)
    fake_mgr = FakeSessionManager(session)
    monkeypatch.setattr(routes, "session_manager", fake_mgr)

    resp = await routes.delete_session("s1", user={"sub": "2"})

    assert fake_mgr.closed == ["s1"]
    assert resp["msg"] == "已删除"


@pytest.mark.asyncio
async def test_delete_session_forbidden(monkeypatch):
    """归属他人 → 403，close_session 不被调用。"""
    session = Session(session_id="s1", user_id=3)
    fake_mgr = FakeSessionManager(session)
    monkeypatch.setattr(routes, "session_manager", fake_mgr)

    with pytest.raises(Exception) as exc_info:
        await routes.delete_session("s1", user={"sub": "2"})
    assert exc_info.value.status_code == 403
    assert fake_mgr.closed == []


@pytest.mark.asyncio
async def test_delete_session_not_found(monkeypatch):
    """会话不存在 → 404。"""
    fake_mgr = FakeSessionManager(None)
    monkeypatch.setattr(routes, "session_manager", fake_mgr)

    with pytest.raises(Exception) as exc_info:
        await routes.delete_session("ghost", user={"sub": "2"})
    assert exc_info.value.status_code == 404
