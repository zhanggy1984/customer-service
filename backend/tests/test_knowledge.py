"""知识库管理单元测试（mock mysql_pool / vector_store / embedder）。

覆盖核心逻辑（kb_store.py）：
- upsert: MySQL 落库 + ChromaDB 重建
- ChromaDB 写失败 → sync_status='pending' + 抛异常（由路由层转 502）
- reconcile_pending: 增量补偿只重建 pending 行
- sync_full: 全量重建 + 孤儿清理
- delete_source: MySQL 删除 + ChromaDB 尽力清理（失败不阻塞）
"""
from unittest.mock import AsyncMock

import pytest

from app.rag import kb_store
from app.rag.interfaces import Document


class FakeMySQLPool:
    """记录调用 + 可配置 fetchall/fetchone 返回值。"""

    def __init__(self):
        self.rows: list[dict] = []
        self.one: dict | None = None
        self.calls: list[tuple] = []

    async def fetchall(self, sql, params=None):
        self.calls.append(("fetchall", sql, params))
        return self.rows

    async def fetchone(self, sql, params=None):
        self.calls.append(("fetchone", sql, params))
        return self.one

    async def execute(self, sql, params=None):
        self.calls.append(("execute", sql, params))
        return 1


class FakeVectorStore:
    """内存版向量库，模拟 ChromaDB 行为。"""

    def __init__(self):
        self.docs: list[Document] = []

    async def delete_by_source(self, source):
        self.docs = [d for d in self.docs if d.metadata["source"] != source]

    async def add_documents(self, docs, embeddings=None):
        self.docs.extend(docs)

    def count(self):
        return len(self.docs)

    def count_by_source(self, source):
        return sum(1 for d in self.docs if d.metadata["source"] == source)

    def get_all(self):
        return list(self.docs)


def _doc(source: str, chunk: int, text: str = "内容") -> Document:
    return Document(id=f"{source}:{chunk}", text=text, metadata={"source": source, "chunk": chunk})


@pytest.fixture
def deps(monkeypatch):
    pool = FakeMySQLPool()
    vs = FakeVectorStore()
    monkeypatch.setattr(kb_store, "mysql_pool", pool)

    async def fake_embed(texts):
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(kb_store.vector_store, "delete_by_source", vs.delete_by_source)
    monkeypatch.setattr(kb_store.vector_store, "add_documents", vs.add_documents)
    monkeypatch.setattr(kb_store.vector_store, "count_by_source", vs.count_by_source)
    monkeypatch.setattr(kb_store.vector_store, "get_all", vs.get_all)
    monkeypatch.setattr(kb_store.embedder, "embed_documents", fake_embed)
    # delete_source/sync_full 都会 _clear_rag_cache → retriever.clear_cache → turn_cache.flush_all，
    # 后者会连真实 Redis（1s 超时）：mock 掉避免单测依赖外部缓存（清缓存逻辑由 test_rag 专测）。
    monkeypatch.setattr(kb_store.retriever, "clear_cache", AsyncMock())
    return pool, vs


@pytest.mark.asyncio
async def test_upsert_writes_mysql_and_chroma(deps):
    pool, vs = deps
    count = await kb_store.upsert("return_policy", "七天无理由退货。", "admin")
    assert count > 0
    # MySQL 已落源数据
    insert_call = [c for c in pool.calls if c[0] == "execute" and "INSERT" in c[1]]
    assert insert_call, "应执行 MySQL INSERT"
    # ChromaDB 已重建该 source 的 chunks
    assert vs.count_by_source("return_policy") == count


@pytest.mark.asyncio
async def test_upsert_chroma_fail_marks_pending(deps, monkeypatch):
    pool, vs = deps

    async def boom(*a, **kw):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(kb_store.vector_store, "add_documents", boom)
    with pytest.raises(RuntimeError):
        await kb_store.upsert("faq", "常见问题回答", "admin")
    # 失败后该行被标记 pending
    pending = [c for c in pool.calls if c[0] == "execute" and "pending" in c[1]]
    assert pending, "ChromaDB 失败后应 UPDATE sync_status='pending'"


@pytest.mark.asyncio
async def test_reconcile_pending_only_rebuilds_pending(deps):
    pool, vs = deps
    pool.rows = [
        {"source": "faq", "content": "退款多久到账？"},
    ]
    done = await kb_store.reconcile_pending()
    assert done == 1
    # 补偿后清标记
    clear = [c for c in pool.calls if c[0] == "execute" and "sync_status='ok'" in c[1]]
    assert clear, "补偿成功应清 pending 标记"
    assert vs.count_by_source("faq") > 0


@pytest.mark.asyncio
async def test_sync_full_rebuilds_and_cleans_orphan(deps):
    pool, vs = deps
    # MySQL 只有 return_policy；ChromaDB 里还有孤儿 after_sales_policy
    pool.rows = [{"source": "return_policy", "content": "七天无理由退货。"}]
    vs.docs = [_doc("return_policy", 0), _doc("after_sales_policy", 0)]
    result = await kb_store.sync_full()
    assert result["synced"] == 1
    assert result["orphan_removed"] == 1
    # 孤儿已被清理，只剩 MySQL 中的文档
    sources = {d.metadata["source"] for d in vs.docs}
    assert sources == {"return_policy"}


@pytest.mark.asyncio
async def test_delete_source_removes_mysql_and_chroma(deps):
    pool, vs = deps
    vs.docs = [_doc("faq", 0)]
    await kb_store.delete_source("faq", "admin")
    deletes = [c for c in pool.calls if c[0] == "execute" and "DELETE FROM knowledge_docs" in c[1]]
    assert deletes, "应删除 MySQL 源数据"
    assert vs.count_by_source("faq") == 0


@pytest.mark.asyncio
async def test_delete_source_chroma_fail_does_not_block(deps, monkeypatch):
    pool, vs = deps

    async def boom(source):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(kb_store.vector_store, "delete_by_source", boom)
    # ChromaDB 清理失败不阻塞删除（孤儿由全量对账兜底）
    await kb_store.delete_source("faq", "admin")
    deletes = [c for c in pool.calls if c[0] == "execute" and "DELETE FROM knowledge_docs" in c[1]]
    assert deletes


# ==================== content_hash 增量跳检 ====================


def _count_add_calls(monkeypatch, vs):
    """包一层 add_documents 计数，返回计数器。"""
    calls = {"n": 0}
    orig = vs.add_documents

    async def counting_add(docs, embeddings=None):
        calls["n"] += 1
        await orig(docs, embeddings)

    monkeypatch.setattr(kb_store.vector_store, "add_documents", counting_add)
    return calls


@pytest.mark.asyncio
async def test_upsert_same_content_skips_rebuild(deps, monkeypatch):
    """同内容重复上传 → 幂等跳过重建（hash 一致 + 向量库在），不重写向量。"""
    pool, vs = deps
    pool.one = {"content_hash": kb_store._content_hash("七天无理由退货。"), "sync_status": "ok"}
    vs.docs = [_doc("return_policy", 0), _doc("return_policy", 1)]
    calls = _count_add_calls(monkeypatch, vs)
    count = await kb_store.upsert("return_policy", "七天无理由退货。", "admin")
    assert calls["n"] == 0  # 未重建
    assert count == 2  # 返回现有 chunk 数


@pytest.mark.asyncio
async def test_upsert_same_content_but_vector_missing_rebuilds(deps, monkeypatch):
    """hash 一致但向量库缺该 source（异常恢复）→ 不得跳检，必须重建。"""
    pool, vs = deps
    pool.one = {"content_hash": kb_store._content_hash("七天无理由退货。"), "sync_status": "ok"}
    # vs.docs 为空 = 向量库丢失
    calls = _count_add_calls(monkeypatch, vs)
    count = await kb_store.upsert("return_policy", "七天无理由退货。", "admin")
    assert calls["n"] == 1
    assert count > 0


@pytest.mark.asyncio
async def test_upsert_changed_content_rebuilds(deps, monkeypatch):
    """内容变更 → 重建，INSERT 带新 hash 回填。"""
    pool, vs = deps
    pool.one = {"content_hash": kb_store._content_hash("旧内容")}
    calls = _count_add_calls(monkeypatch, vs)
    await kb_store.upsert("return_policy", "七天无理由退货。", "admin")
    assert calls["n"] == 1
    insert = [c for c in pool.calls if c[0] == "execute" and "INSERT" in c[1]]
    assert insert
    # hash 走参数化传参（params 而非拼进 SQL 文本）
    assert kb_store._content_hash("七天无理由退货。") in insert[0][2]


@pytest.mark.asyncio
async def test_sync_full_skips_unchanged(deps, monkeypatch):
    """内容未变且向量库在 → 跳过重建（skipped），不重写向量。"""
    pool, vs = deps
    ch = kb_store._content_hash("七天无理由退货。")
    pool.rows = [{"source": "return_policy", "content": "七天无理由退货。", "content_hash": ch, "sync_status": "ok"}]
    vs.docs = [_doc("return_policy", 0)]
    calls = _count_add_calls(monkeypatch, vs)
    result = await kb_store.sync_full()
    assert calls["n"] == 0
    assert result["synced"] == 0 and result["skipped"] == 1


@pytest.mark.asyncio
async def test_sync_full_force_rebuilds(deps, monkeypatch):
    """force=True → 全量重建，即便内容未变。"""
    pool, vs = deps
    ch = kb_store._content_hash("七天无理由退货。")
    pool.rows = [{"source": "return_policy", "content": "七天无理由退货。", "content_hash": ch, "sync_status": "ok"}]
    vs.docs = [_doc("return_policy", 0)]
    calls = _count_add_calls(monkeypatch, vs)
    result = await kb_store.sync_full(force=True)
    assert calls["n"] == 1
    assert result["synced"] == 1 and result["skipped"] == 0


@pytest.mark.asyncio
async def test_sync_full_null_hash_rebuilds(deps, monkeypatch):
    """存量行 hash 为 NULL → 升级后首次 sync 全量重建并回填。"""
    pool, vs = deps
    pool.rows = [{"source": "return_policy", "content": "七天无理由退货。"}]  # 无 content_hash
    vs.docs = [_doc("return_policy", 0)]
    calls = _count_add_calls(monkeypatch, vs)
    result = await kb_store.sync_full()
    assert calls["n"] == 1
    assert result["synced"] == 1 and result["skipped"] == 0
    # 回填 hash：UPDATE 语句带新 hash
    updates = [c for c in pool.calls if c[0] == "execute" and "content_hash" in c[1] and "UPDATE" in c[1]]
    assert updates


@pytest.mark.asyncio
async def test_seed_writes_content_hash(deps, tmp_path, monkeypatch):
    """seed 落库带 content_hash：后续 sync 可跳检。"""
    pool, vs = deps
    (tmp_path / "return_policy.md").write_text("七天无理由退货。", encoding="utf-8")
    monkeypatch.setattr(kb_store, "KNOWLEDGE_DIR", tmp_path)
    await kb_store._seed_from_markdown()
    insert = [c for c in pool.calls if c[0] == "execute" and "INSERT" in c[1]]
    assert insert
    # hash 走参数化传参（params 而非拼进 SQL 文本）
    assert kb_store._content_hash("七天无理由退货。") in insert[0][2]


@pytest.mark.asyncio
async def test_upsert_pending_never_skips(deps, monkeypatch):
    """sync_status='pending'（上次重建失败残留）→ 不得跳检，必须重建保补偿。"""
    pool, vs = deps
    pool.one = {"content_hash": kb_store._content_hash("七天无理由退货。"), "sync_status": "pending"}
    vs.docs = [_doc("return_policy", 0), _doc("return_policy", 1)]  # 残留的部分向量
    calls = _count_add_calls(monkeypatch, vs)
    await kb_store.upsert("return_policy", "七天无理由退货。", "admin")
    assert calls["n"] == 1  # pending 行不跳检，重建修复


@pytest.mark.asyncio
async def test_sync_full_pending_never_skipped(deps, monkeypatch):
    """pending 行即使 hash 匹配也不跳检——否则把"待补偿"洗成 ok，自愈链路断。"""
    pool, vs = deps
    ch = kb_store._content_hash("七天无理由退货。")
    pool.rows = [{"source": "return_policy", "content": "七天无理由退货。",
                  "content_hash": ch, "sync_status": "pending"}]
    vs.docs = [_doc("return_policy", 0)]  # 残留的部分向量
    calls = _count_add_calls(monkeypatch, vs)
    result = await kb_store.sync_full()
    assert calls["n"] == 1  # 重建，而非跳过
    assert result["synced"] == 1 and result["skipped"] == 0
    # pending 行重建后回填 hash，未被跳检分支误洗


def test_content_hash_binds_embedding_model(monkeypatch):
    """hash 绑定 embedding 模型版本：换模型后旧 hash 全部失效 → sync 自动全量重建。"""
    h0 = kb_store._content_hash("七天无理由退货。")
    monkeypatch.setattr(kb_store.settings, "embedding_model", "other-embed-model")
    assert kb_store._content_hash("七天无理由退货。") != h0
