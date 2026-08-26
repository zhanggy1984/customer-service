"""policy_tools.search_policy 单测（mock retriever.search）。

覆盖 FC 契约：入口 query 清洗（normalize_query + 超长截断 + 空回退）、
信封结构（data.results/source_count/max_score 聚合字段）。
"""
from unittest.mock import AsyncMock

import pytest

from app.agent.function_calling.tools import policy_tools
from app.rag.interfaces import SearchResult


@pytest.mark.asyncio
async def test_search_policy_envelope(monkeypatch):
    """成功检索 → 信封结构 + 聚合字段（source_count/max_score 供决策 LLM 判断）。"""
    results = [
        SearchResult(id="1", text="退货时限 7 天", score=0.91,
                     metadata={"source": "return_policy", "heading_path": ["退货时限"]}),
        SearchResult(id="2", text="非质量问题运费自理", score=0.42, metadata={"source": "return_policy"}),
    ]
    monkeypatch.setattr(policy_tools.retriever, "search", AsyncMock(return_value=results))
    out = await policy_tools.search_policy({"query": "退货政策是什么"}, 1, "s")
    assert out["ok"] is True
    assert out["error"] is None
    assert len(out["data"]["results"]) == 2
    assert out["data"]["source_count"] == 2
    assert out["data"]["max_score"] == 0.91  # 聚合置信信号
    # 溯源标签透出（source > heading_path）
    assert out["data"]["results"][0]["source"] == "return_policy > 退货时限"


@pytest.mark.asyncio
async def test_search_policy_query_cleaned(monkeypatch):
    """query 入口清洗：客套前缀/尾部标点剥离后再检索（与缓存归一化同口径）。"""
    received = {}

    async def fake_search(query):
        received["query"] = query
        return []

    monkeypatch.setattr(policy_tools.retriever, "search", fake_search)
    await policy_tools.search_policy({"query": "请问，退货时限几天？"}, 1, "s")
    assert received["query"] == "退货时限几天"


@pytest.mark.asyncio
async def test_search_policy_query_truncated(monkeypatch):
    """超长 query → 前缀截断到 _QUERY_MAX_LEN（防整段对话拖垮召回）。"""
    received = {}

    async def fake_search(query):
        received["query"] = query
        return []

    monkeypatch.setattr(policy_tools.retriever, "search", fake_search)
    long_q = "退" * (policy_tools._QUERY_MAX_LEN + 50)
    await policy_tools.search_policy({"query": long_q}, 1, "s")
    assert len(received["query"]) == policy_tools._QUERY_MAX_LEN


@pytest.mark.asyncio
async def test_search_policy_empty_query_clean_fallback(monkeypatch):
    """纯客套 query 归一化后为空 → 回退原文（防空 query 拖垮召回）。"""
    received = {}

    async def fake_search(query):
        received["query"] = query
        return []

    monkeypatch.setattr(policy_tools.retriever, "search", fake_search)
    await policy_tools.search_policy({"query": "谢谢"}, 1, "s")  # normalize("谢谢")=="" → 回退原文
    assert received["query"] == "谢谢"


@pytest.mark.asyncio
async def test_search_policy_missing_query_error(monkeypatch):
    """缺 query → 统一错误信封（missing_query）。"""
    out = await policy_tools.search_policy({}, 1, "s")
    assert out["ok"] is False
    assert out["data"] is None
    assert out["error"]["code"] == "missing_query"


@pytest.mark.asyncio
async def test_search_policy_retrieval_unavailable(monkeypatch):
    """检索故障（Milvus/embedding 挂）→ retrieval_unavailable 错误码，不伪装"空"（未收录）。"""
    from app.rag.retriever import RetrievalUnavailableError

    monkeypatch.setattr(
        policy_tools.retriever, "search",
        AsyncMock(side_effect=RetrievalUnavailableError("知识库检索暂不可用")),
    )
    out = await policy_tools.search_policy({"query": "退货政策"}, 1, "s")
    assert out["ok"] is False
    assert out["data"] is None
    assert out["error"]["code"] == "retrieval_unavailable"
