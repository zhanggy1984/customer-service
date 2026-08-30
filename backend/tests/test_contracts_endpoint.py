"""契约清单端点单测：GET /api/contracts 返回评测平台可自动发现的接口与场景清单。

覆盖 contracts.py 的 MANIFEST 结构：chat 为唯一 llm=true 的 SSE 接口，
scenes 覆盖四个评测场景（greeting/order_query/after_sales/human_handoff），
contract 段（manifest v2）为平台驱动本 agent 的权威声明。
"""
import asyncio

import pytest

from app.api.contracts import contracts


@pytest.mark.asyncio
async def test_contracts_manifest_structure():
    m = await contracts()
    assert m["agent"] == "customer-service"
    assert m["contract_version"] == "2.0"

    # chat 是唯一 llm=true 的 SSE 接口，路径与平台 seed 快照一致
    llm_ifaces = [i for i in m["interfaces"] if i.get("llm")]
    assert len(llm_ifaces) == 1
    chat = llm_ifaces[0]
    assert chat["name"] == "chat"
    assert chat["contract_type"] == "sse"
    assert chat["path"] == "/api/v1/sessions/{sid}/messages"
    assert chat["method"] == "POST"

    # 辅助接口（登录）不进入 agent_interface
    login = [i for i in m["interfaces"] if i["name"] == "login"]
    assert login and login[0]["llm"] is False

    # 场景清单覆盖四个评测场景
    tags = {s["tag"] for s in m["scenes"]}
    assert {"greeting", "order_query", "after_sales", "human_handoff"} <= tags

    # contract 段（manifest v2）：驱动契约必须带 login/session 两个 prepare + request
    c = m["contract"]
    assert c["type"] == "sse"
    assert [p["name"] for p in c["prepare"]] == ["login", "session"]
    assert c["request"]["method"] == "POST"
    assert "{{input.content}}" in str(c)
