"""端到端集成测试：完整退货流程（真实服务 + DeepSeek + DB）。

依赖 docker compose 服务运行（localhost:8000），不可用时跳过。
"""
import asyncio
import json

import httpx
import pytest

from tests.conftest import SERVICE_READY

BASE = "http://localhost:8000/api/v1"
pytestmark = pytest.mark.skipif(not SERVICE_READY, reason="服务未运行")


def _parse_sse(body: bytes) -> list[dict]:
    events = []
    for line in body.decode("utf-8").split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[5:]))
    return events


@pytest.fixture
async def user():
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{BASE}/auth/login", json={"username": "user_1", "password": "123456"})
        assert resp.status_code == 200, resp.text
        token = resp.json()["access_token"]
        yield {"client": client, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.mark.asyncio
async def test_full_return_flow(user):
    client: httpx.AsyncClient = user["client"]
    headers = user["headers"]

    # 幂等化：重置 ORD-20240801-001 的退货残留。
    # 完整退货流程会创建退单并把 SKU 置 RETURNED，不清理则二次运行判定"已退过"。
    # 用业务订单号定位 DB id：admin reset 会删光重插种子订单，自增 id 漂移，硬编码 id=1 会匹配不到。
    from app.infrastructure.mysql import mysql_pool

    await mysql_pool.init()
    await mysql_pool.execute(
        "DELETE FROM return_orders WHERE order_id IN (SELECT id FROM orders WHERE order_id=%s)",
        ("ORD-20240801-001",),
    )
    await mysql_pool.execute(
        "UPDATE order_items SET status='NORMAL' WHERE order_id IN (SELECT id FROM orders WHERE order_id=%s)",
        ("ORD-20240801-001",),
    )

    resp = await client.post(f"{BASE}/sessions", headers=headers)
    sid = resp.json()["session_id"]

    # 轮 1: 我要退货 → 追问原因
    async with client.stream("POST", f"{BASE}/sessions/{sid}/messages", headers=headers,
                             json={"content": "我要退货 ORD-20240801-001"}) as r:
        assert r.status_code == 200
        body = await r.aread()
    done = [e for e in _parse_sse(body) if e["type"] == "done"][-1]
    assert "退货原因" in done["content"], done["content"]

    # 轮 2: 质量问题 → 确认信息
    async with client.stream("POST", f"{BASE}/sessions/{sid}/messages", headers=headers,
                             json={"content": "质量问题"}) as r:
        body = await r.aread()
    done = [e for e in _parse_sse(body) if e["type"] == "done"][-1]
    assert "确认" in done["content"], done["content"]

    # 轮 3: 确认 → 退单号 RC-
    async with client.stream("POST", f"{BASE}/sessions/{sid}/messages", headers=headers,
                             json={"content": "确认"}) as r:
        body = await r.aread()
    done = [e for e in _parse_sse(body) if e["type"] == "done"][-1]
    assert "RC-" in done["content"], done["content"]


@pytest.mark.asyncio
async def test_order_status(user):
    client = user["client"]
    headers = user["headers"]

    resp = await client.post(f"{BASE}/sessions", headers=headers)
    sid = resp.json()["session_id"]

    async with client.stream("POST", f"{BASE}/sessions/{sid}/messages", headers=headers,
                             json={"content": "查一下订单 ORD-20240805-002"}) as r:
        body = await r.aread()
    done = [e for e in _parse_sse(body) if e["type"] == "done"][-1]
    assert "已发货" in done["content"], done["content"]


@pytest.mark.asyncio
async def test_policy_rag(user):
    client = user["client"]
    headers = user["headers"]

    resp = await client.post(f"{BASE}/sessions", headers=headers)
    sid = resp.json()["session_id"]

    async with client.stream("POST", f"{BASE}/sessions/{sid}/messages", headers=headers,
                             json={"content": "退货时限是多久"}) as r:
        body = await r.aread()
    done = [e for e in _parse_sse(body) if e["type"] == "done"][-1]
    # LLM 措辞可能为 "7 天"（带空格）或 "7天内"（无空格），断言放宽兼容两者
    assert ("7 天" in done["content"] or "7天内" in done["content"]), done["content"]
