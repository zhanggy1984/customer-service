"""知识库文档管理：MySQL 为源（source of truth），ChromaDB 为派生向量索引。

数据流原则：
- MySQL knowledge_docs 存原始 markdown 全文（权威）；ChromaDB 只存分块向量快照（派生）。
- 任何修改都从 MySQL 的 content 重新分块生成 ChromaDB chunks，ChromaDB 不独立演进，
  因此拼接还原原始文档的问题（overlap 重复 / chunk 被改过）从根上消失。
- 写 ChromaDB 失败 → 该行 sync_status='pending'，由增量补偿（自动）或全量对账（手动）自愈。

一致性策略（避免全量对账成为常态路径）：
- 增量补偿：每次写操作后补偿 pending 行，只重建失败的那几篇，成本 O(失败数)。
- 全量对账：仅 admin 手动触发（POST /admin/knowledge/sync）或 ChromaDB 数据丢失时
  启动恢复，一次重建全部 + 清理孤儿 chunks。不进每次操作的常态路径。
"""
import hashlib
from pathlib import Path

from app.infrastructure.mysql import mysql_pool
from app.rag import vector_store
from app.rag.embedder import embedder
from app.rag.interfaces import Document
from app.rag.retriever import retriever
from app.rag.splitter import chunk_document
from app.utils.logger import logger

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


async def _clear_rag_cache() -> None:
    """清 RAG 精确缓存（更新后必须清，否则 600s 内命中旧内容）。Redis 不可用不阻塞同步。"""
    try:
        await retriever.clear_cache()
    except Exception as exc:
        logger.warning("event=kb_clear_cache_fail error=%s", str(exc))


def _make_docs(source: str, content: str) -> list[Document]:
    """文档内容结构化切分生成 Document（doc_id = md5(source:chunk) 保证可寻址）。

    chunk_document 返回带 heading_path/section_id 等 metadata 的块，这里合并 source
    （kb_store 层概念）后透传给向量库；检索溯源与章节扩充均依赖这些字段。
    """
    docs = []
    for c in chunk_document(content, source):
        text = c["content"]
        if not text.strip():
            continue
        meta = {"source": source, **c["metadata"]}
        doc_id = hashlib.md5(f"{source}:{c['metadata']['chunk_index']}".encode()).hexdigest()
        docs.append(Document(id=doc_id, text=text, metadata=meta))
    return docs


async def rebuild_source(source: str, content: str) -> int:
    """重建一篇文档的 ChromaDB chunks（删旧 → embedding → 写新）。返回 chunk 数。"""
    docs = _make_docs(source, content)
    if not docs:
        raise ValueError(f"文档内容为空: {source}")
    await vector_store.delete_by_source(source)
    embeddings = await embedder.embed_documents([d.text for d in docs])
    await vector_store.add_documents(docs, embeddings)
    await _clear_rag_cache()
    return len(docs)


async def list_docs() -> list[dict]:
    """文档级列表：标题 / 内容摘要 / 块数 / 同步状态 / 更新时间 / 操作人。"""
    rows = await mysql_pool.fetchall(
        "SELECT source, content, sync_status, updated_by, created_at, updated_at "
        "FROM knowledge_docs ORDER BY updated_at DESC"
    )
    for row in rows:
        row["chunk_count"] = vector_store.count_by_source(row["source"])
    return rows


async def upsert(source: str, content: str, updated_by: str) -> int:
    """上传或覆盖一篇文档。

    MySQL 先落（权威），ChromaDB 同步失败 → sync_status='pending' 并抛异常，
    由调用方返回 502；admin 重试或全量对账自愈。
    """
    await mysql_pool.execute(
        "INSERT INTO knowledge_docs (source, content, updated_by, sync_status) "
        "VALUES (%s, %s, %s, 'ok') "
        "ON DUPLICATE KEY UPDATE content=VALUES(content), updated_by=VALUES(updated_by), sync_status='ok'",
        (source, content, updated_by),
    )
    try:
        chunk_count = await rebuild_source(source, content)
    except Exception:
        await _mark_pending(source)
        raise
    logger.info(
        "event=kb_upsert source=%s chunks=%s by=%s", source, chunk_count, updated_by
    )
    return chunk_count


async def update_source(source: str, content: str, updated_by: str) -> int:
    """修改一篇已有文档，语义同 upsert（内容级更新）。"""
    return await upsert(source, content, updated_by)


async def delete_source(source: str, updated_by: str) -> None:
    """删除一篇文档。

    以 MySQL 为权威：文档从源删除即视为删除完成。ChromaDB 清理尽力而为，
    失败仅记日志（残留 chunks 由全量对账的孤儿清理兜底），不阻塞删除。
    """
    cur = await mysql_pool.execute("DELETE FROM knowledge_docs WHERE source=%s", (source,))
    if cur == 0:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="文档不存在")
    try:
        await vector_store.delete_by_source(source)
    except Exception as exc:
        logger.error("event=kb_delete_chroma_fail source=%s error=%s", source, str(exc))
    logger.info("event=kb_delete source=%s by=%s", source, updated_by)


async def _mark_pending(source: str) -> None:
    await mysql_pool.execute(
        "UPDATE knowledge_docs SET sync_status='pending' WHERE source=%s", (source,)
    )


async def _pending_sources() -> list[dict]:
    return await mysql_pool.fetchall(
        "SELECT source, content FROM knowledge_docs WHERE sync_status='pending'"
    )


async def reconcile_pending() -> int:
    """增量补偿：只重建 sync_status='pending' 的文档，成功清标记。

    每次知识库写操作后调用。成本跟随失败量，不扫描全量。
    """
    pending = await _pending_sources()
    done = 0
    for row in pending:
        try:
            await rebuild_source(row["source"], row["content"])
            await mysql_pool.execute(
                "UPDATE knowledge_docs SET sync_status='ok' WHERE source=%s",
                (row["source"],),
            )
            done += 1
        except Exception as exc:
            logger.error(
                "event=kb_reconcile_pending_fail source=%s error=%s", row["source"], str(exc)
            )
    if done:
        logger.info("event=kb_reconcile_pending done=%s", done)
    return done


async def sync_full() -> dict:
    """全量对账（admin 手动按钮 / ChromaDB 丢失恢复）。

    1. 每篇 MySQL 文档重建 ChromaDB chunks（同时把 pending 行拉回 ok）。
    2. 清理孤儿：ChromaDB 中存在但 MySQL 中已删除的 source。
    不进常态路径，仅异常恢复时调用，避免全量扫描压力常态化。
    """
    rows = await mysql_pool.fetchall("SELECT source, content FROM knowledge_docs")
    synced = 0
    for row in rows:
        try:
            await rebuild_source(row["source"], row["content"])
            await mysql_pool.execute(
                "UPDATE knowledge_docs SET sync_status='ok' WHERE source=%s",
                (row["source"],),
            )
            synced += 1
        except Exception as exc:
            logger.error(
                "event=kb_sync_full_fail source=%s error=%s", row["source"], str(exc)
            )
    # 孤儿清理：ChromaDB 的 source 集合 - MySQL 的 source 集合
    mysql_sources = {r["source"] for r in rows}
    chroma_sources = {d.metadata.get("source", "") for d in vector_store.get_all()}
    orphan_removed = 0
    for source in sorted(chroma_sources - mysql_sources):
        if not source:
            continue
        try:
            await vector_store.delete_by_source(source)
            orphan_removed += 1
        except Exception as exc:
            logger.error(
                "event=kb_orphan_clean_fail source=%s error=%s", source, str(exc)
            )
    logger.info(
        "event=kb_sync_full synced=%s orphan_removed=%s", synced, orphan_removed
    )
    return {"synced": synced, "orphan_removed": orphan_removed}


async def ensure_init() -> None:
    """启动时初始化（替代原 rag/init.init_knowledge 的独立写库逻辑）。

    - MySQL 表空 → 从 knowledge/*.md 灌入（source=文件名, updated_by='system'）。
    - ChromaDB 空而 MySQL 有数据 → 全量重建（ChromaDB 数据丢失恢复）。
    - 否则仅补偿 pending 行（增量），不碰已同步文档。
    """
    total = await mysql_pool.fetchone("SELECT COUNT(*) AS c FROM knowledge_docs")
    mysql_count = (total or {}).get("c", 0)
    try:
        chroma_count = vector_store.count()
    except Exception as exc:
        logger.warning("event=kb_init_chroma_unreachable error=%s", str(exc))
        chroma_count = -1  # 未知 → 只做增量补偿，不重建

    if mysql_count == 0:
        await _seed_from_markdown()
        await sync_full()
        return

    if chroma_count == 0:
        # MySQL 有数据但向量库空 → 判定为向量数据丢失，全量重建
        logger.warning("event=kb_init_chroma_empty trigger_full_rebuild")
        await sync_full()
        return

    await reconcile_pending()


async def _seed_from_markdown() -> None:
    """首次启动：从 knowledge/*.md 把文档全文灌入 MySQL（升级部署时同样走这里）。"""
    if not KNOWLEDGE_DIR.exists():
        logger.warning("event=kb_seed_dir_missing dir=%s", KNOWLEDGE_DIR)
        return
    count = 0
    for md_path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        content = md_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        await mysql_pool.execute(
            "INSERT IGNORE INTO knowledge_docs (source, content, updated_by, sync_status) "
            "VALUES (%s, %s, 'system', 'ok')",
            (md_path.stem, content),
        )
        count += 1
    logger.info("event=kb_seed_from_markdown docs=%s", count)
