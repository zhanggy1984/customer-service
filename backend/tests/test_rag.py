"""RAG 检索单元测试（mock embedder/vector_store/redis）。"""
import pytest

from app.rag.interfaces import SearchResult
from app.rag.retriever import Retriever
from app.rag.retriever import embedder as embedder_mod
from app.rag.retriever import reranker as reranker_mod
from app.rag.retriever import vector_store as vs_mod


async def fake_rerank(query, results):
    """单测不加载 bge-reranker 模型，重排退化为原序。"""
    return results


class FakeRedis:
    def __init__(self, cache=None):
        self._cache = cache

    async def get(self, key):
        return self._cache

    async def set(self, *a, **kw):
        return None

    async def scan_iter(self, match=None):
        return iter(())


@pytest.mark.asyncio
async def test_search_hit(monkeypatch):
    r = Retriever()
    r._redis = FakeRedis()

    async def fake_embed(text):
        return [0.1] * 10

    async def fake_search(vec, top_k):
        return [SearchResult(id="1", text="退货时限 7 天", score=0.9, metadata={"source": "return_policy"})]

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)

    results = await r.search("退货时限")
    assert len(results) == 1
    assert results[0].score >= 0.3


@pytest.mark.asyncio
async def test_search_empty_below_threshold(monkeypatch):
    r = Retriever()
    r._redis = FakeRedis()

    async def fake_embed(text):
        return [0.1] * 10

    async def fake_search(vec, top_k):
        return [SearchResult(id="1", text="无关内容", score=0.1, metadata={})]

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)

    results = await r.search("量子物理")
    assert results == []  # score < 0.3 过滤


@pytest.mark.asyncio
async def test_search_cache_hit(monkeypatch):
    import json as _json

    cached = _json.dumps([{"id": "1", "text": "缓存内容", "score": 0.9, "metadata": {"source": "faq"}}], ensure_ascii=False)
    r = Retriever()
    r._redis = FakeRedis(cache=cached)

    results = await r.search("退款多久到账")
    assert len(results) == 1
    assert results[0].text == "缓存内容"
