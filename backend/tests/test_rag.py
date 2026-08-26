"""RAG 检索单元测试（mock embedder/vector_store/redis）。"""
import pytest

from app.rag.interfaces import SearchResult
from app.rag.retriever import RetrievalUnavailableError, Retriever
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
        if False:
            yield  # async generator：供 clear_cache 的 async for 消费（空迭代）
        return


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
async def test_search_embedding_error_raises_retrieval_unavailable(monkeypatch):
    """embedding 失败 ≠ 检索空：抛 RetrievalUnavailableError（供 search_policy 映射故障码）。"""
    r = Retriever()
    r._redis = FakeRedis()

    async def fake_embed(text):
        raise RuntimeError("embedding 模型不可用")

    async def fake_search(vec, top_k):
        raise AssertionError("embedding 失败时不应走到向量检索")

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)

    with pytest.raises(RetrievalUnavailableError):
        await r.search("退货时限")


@pytest.mark.asyncio
async def test_search_vector_store_error_raises_retrieval_unavailable(monkeypatch):
    """Milvus 查询失败 ≠ 检索空：抛 RetrievalUnavailableError（不伪装成空列表）。"""
    r = Retriever()
    r._redis = FakeRedis()

    async def fake_embed(text):
        return [0.1] * 10

    async def fake_search(vec, top_k):
        raise RuntimeError("Milvus connection refused")

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)

    with pytest.raises(RetrievalUnavailableError):
        await r.search("退货时限")


@pytest.mark.asyncio
async def test_search_cache_hit(monkeypatch):
    import json as _json

    cached = _json.dumps([{"id": "1", "text": "缓存内容", "score": 0.9, "metadata": {"source": "faq"}}], ensure_ascii=False)
    r = Retriever()
    r._redis = FakeRedis(cache=cached)

    results = await r.search("退款多久到账")
    assert len(results) == 1
    assert results[0].text == "缓存内容"


@pytest.mark.asyncio
async def test_clear_cache_flushes_turn_cache(monkeypatch):
    """clear_cache 联动清回合缓存：回合答案派生自检索文档，与 RAG 精确缓存同源失效。"""
    from app.infrastructure import turn_cache as tc

    flushed = {"n": 0}

    async def fake_flush():
        flushed["n"] += 1

    monkeypatch.setattr(tc, "flush_all", fake_flush)
    r = Retriever()
    r._redis = FakeRedis()
    await r.clear_cache()
    assert flushed["n"] == 1  # 清 RAG 缓存后必须一并 flush 回合缓存


@pytest.mark.asyncio
async def test_search_query_normalized(monkeypatch):
    """检索 query 归一化：剥客套/全角/尾部标点后再 embed 与建缓存 key（与文档侧清洗同口径）。"""
    r = Retriever()
    r._redis = FakeRedis()
    received = {}

    async def fake_embed(text):
        received["query"] = text
        return [0.1] * 10

    async def fake_search(vec, top_k):
        return [SearchResult(id="1", text="退货时限 7 天", score=0.9, metadata={"source": "return_policy"})]

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)

    await r.search("请问退货时限？")
    assert received["query"] == "退货时限"  # 剥客套前缀 + 尾部标点


@pytest.mark.asyncio
async def test_search_query_empty_fallback_to_original(monkeypatch):
    """纯客套 query 归一化后为空 → 回退原文，防空 query 拖垮召回。"""
    r = Retriever()
    r._redis = FakeRedis()
    received = {}

    async def fake_embed(text):
        received["query"] = text
        return [0.1] * 10

    async def fake_search(vec, top_k):
        return []

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)

    await r.search("谢谢")
    assert received["query"] == "谢谢"  # normalize 后为空，回退原文


def test_source_label_with_page_num():
    """page_num 透传时溯源标签追加页码（PDF 接入后生效，当前 markdown 无该字段）。"""
    from app.rag.interfaces import source_label

    assert (
        source_label({"source": "policy.pdf", "heading_path": ["第三章"], "page_num": 42})
        == "policy.pdf > 第三章（第 42 页）"
    )
    assert source_label({"source": "policy.pdf", "page_num": 42}) == "policy.pdf（第 42 页）"
    # 无 page_num 不产生页码（现有 markdown 数据路径不变）
    assert (
        source_label({"source": "return_policy", "heading_path": ["退货时限"]})
        == "return_policy > 退货时限"
    )


@pytest.mark.asyncio
async def test_search_expands_section_siblings(monkeypatch):
    """章节扩充：命中 chunk 的同 section 兄弟 chunk 合并进 context。"""
    from app.rag.interfaces import Document

    r = Retriever()
    r._redis = FakeRedis()

    async def fake_embed(text):
        return [0.1] * 10

    async def fake_search(vec, top_k):
        return [
            SearchResult(
                id="c2", text="非质量问题运费自理。", score=0.9,
                metadata={"source": "return_policy", "section_id": "return_policy:1",
                          "heading_path": ["退换货政策", "运费规则"], "chunk_index": 1},
            )
        ]

    async def fake_get_all():
        return [
            Document(id="c1", text="退货运费由买家承担。",
                     metadata={"source": "return_policy", "section_id": "return_policy:1",
                               "heading_path": ["退换货政策", "运费规则"], "chunk_index": 0}),
            Document(id="c3", text="签收后 7 天内可退。",
                     metadata={"source": "return_policy", "section_id": "return_policy:2",
                               "heading_path": ["退换货政策", "退货时限"], "chunk_index": 2}),
        ]

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)
    monkeypatch.setattr(vs_mod, "get_all", fake_get_all)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)

    results = await r.search("运费谁出")
    ids = {res.id for res in results}
    assert ids == {"c1", "c2"}  # c3 属其他 section 不扩充
    # 扩充 chunk：score 复用命中分数（不引入 0），且打 is_expanded 标记供观测区分
    c1 = next(x for x in results if x.id == "c1")
    assert c1.score == 0.9
    assert c1.metadata.get("is_expanded") is True
    c2 = next(x for x in results if x.id == "c2")
    assert c2.metadata.get("is_expanded") is None  # 真命中无标记


@pytest.mark.asyncio
async def test_search_section_expand_degraded_without_section_id(monkeypatch):
    """旧数据无 section_id → 不触发扩充，get_all 不被调用。"""
    r = Retriever()
    r._redis = FakeRedis()

    async def fake_embed(text):
        return [0.1] * 10

    async def fake_search(vec, top_k):
        return [SearchResult(id="1", text="旧数据", score=0.9, metadata={"source": "faq"})]

    calls = {"get_all": 0}

    async def fake_get_all():
        calls["get_all"] += 1
        return []

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)
    monkeypatch.setattr(vs_mod, "get_all", fake_get_all)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)

    results = await r.search("旧数据")
    assert len(results) == 1
    assert calls["get_all"] == 0


@pytest.mark.asyncio
async def test_search_section_expand_total_limit(monkeypatch):
    """扩充总量上限：同 section 兄弟很多时 context 不超 _SECTION_EXPAND_TOTAL。"""
    from app.rag.interfaces import Document

    r = Retriever()
    r._redis = FakeRedis()

    async def fake_embed(text):
        return [0.1] * 10

    async def fake_search(vec, top_k):
        return [
            SearchResult(id="hit", text="命中", score=0.9,
                         metadata={"source": "s", "section_id": "s:0"}),
        ]

    async def fake_get_all():
        return [
            Document(id=f"sib-{i}", text=f"兄弟 {i}",
                     metadata={"source": "s", "section_id": "s:0"})
            for i in range(20)
        ]

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)
    monkeypatch.setattr(vs_mod, "get_all", fake_get_all)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)

    results = await r.search("章节")
    assert len(results) <= 6  # _SECTION_EXPAND_TOTAL


@pytest.mark.asyncio
async def test_search_section_expand_prefers_nearby_chunk(monkeypatch):
    """邻近度排序：兄弟 chunk 按与命中 chunk 的 chunk_index 距离补齐，近者优先。"""
    from app.rag.interfaces import Document

    r = Retriever()
    r._redis = FakeRedis()

    async def fake_embed(text):
        return [0.1] * 10

    async def fake_search(vec, top_k):
        return [
            SearchResult(id="hit", text="命中", score=0.8,
                         metadata={"source": "s", "section_id": "s:0", "chunk_index": 3}),
        ]

    async def fake_get_all():
        # 兄弟 chunk 0/1/4/5（命中 3），距命中分别为 3/2/1/2
        return [
            Document(id=f"c{i}", text=f"块{i}",
                     metadata={"source": "s", "section_id": "s:0", "chunk_index": i})
            for i in (0, 1, 4, 5)
        ]

    monkeypatch.setattr(embedder_mod, "embed_query", fake_embed)
    monkeypatch.setattr(vs_mod, "search", fake_search)
    monkeypatch.setattr(vs_mod, "get_all", fake_get_all)
    monkeypatch.setattr(reranker_mod, "rerank", fake_rerank)

    results = await r.search("章节")
    expanded = [x.id for x in results if x.metadata.get("is_expanded")]
    assert expanded[0] == "c4"  # 距命中最近的兄弟最先
    # 近邻(c4，距1) 排在远邻(c0，距3) 之前
    assert expanded.index("c4") < expanded.index("c0")
