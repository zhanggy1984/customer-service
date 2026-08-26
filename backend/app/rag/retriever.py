"""RAG 检索器。

流程: query → embedding → 向量库检索 Top-10 → 交叉编码重排 Top-3 → score≥0.3 过滤 → 章节级扩充。
向量库实现为 MilvusVectorStore（chroma 已移除），此处只面向 IVectorStore 接口。
缓存: Redis L1 精确缓存 rag_cache:{md5}，TTL 600s。
空结果兜底: score < 0.3 或 0 条 → 返回空列表，上层不注入 prompt。
"""
import hashlib
import json

import redis.asyncio as aioredis

from app.config import settings
from app.rag import vector_store
from app.rag.embedder import embedder
from app.rag.interfaces import SearchResult
from app.rag.reranker import reranker
from app.utils.logger import logger

TOP_K = 10
RE_RANK_K = 3
SCORE_THRESHOLD = 0.3
# 章节级扩充后 context 总量上限：防大 section 撑爆 prompt
_SECTION_EXPAND_TOTAL = 6


class Retriever:
    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def init(self) -> None:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def clear_cache(self) -> None:
        """清除全部 RAG 精确缓存（Admin 增删知识库后调用，防止旧缓存污染新检索）。"""
        async for key in self._redis.scan_iter(match="rag_cache:*"):
            await self._redis.delete(key)

    async def search(self, query: str) -> list[SearchResult]:
        cache_key = f"rag_cache:{hashlib.md5(query.encode()).hexdigest()}"

        # L1 精确缓存（Redis 不可用时跳过缓存直接检索）
        try:
            cached = await self._redis.get(cache_key)
        except Exception:
            cached = None
        if cached:
            data = json.loads(cached)
            logger.info("event=rag_cache_hit:L1", extra={"query_len": len(query), "count": len(data)})
            return [SearchResult(**item) for item in data]

        # 检索 + 交叉编码重排（bge-reranker 真重排；失败降级按相似度排序，不阻断检索）
        query_vec = await embedder.embed_query(query)
        results = await vector_store.search(query_vec, top_k=TOP_K)
        try:
            results = await reranker.rerank(query, results)
        except Exception:
            logger.warning("event=rag_rerank_fallback", extra={"query_len": len(query)})
            results = sorted(results, key=lambda r: r.score, reverse=True)[:RE_RANK_K]
        results = [r for r in results if r.score >= SCORE_THRESHOLD]

        if not results:
            logger.info("event=rag_empty", extra={"query_len": len(query), "top_k": TOP_K})
            return []

        # 章节级扩充：合并同 section 兄弟 chunk。长文档内答案常横跨同章节多个 chunk，
        # 精排 top-k 只覆盖片段；扩充后 LLM context 覆盖整节（精排 top 仍在前）。
        results = await self._expand_section_context(results)

        # 回写缓存（Redis 不可用时静默跳过）
        try:
            payload = [
                {"id": r.id, "text": r.text, "score": r.score, "metadata": r.metadata}
                for r in results
            ]
            await self._redis.set(cache_key, json.dumps(payload, ensure_ascii=False), ex=settings.rag_cache_ttl)
        except Exception:
            pass
        logger.info(
            "event=rag_cache_miss",
            extra={"query_len": len(query), "top_score": round(results[0].score, 3), "count": len(results)},
        )
        return results

    async def _expand_section_context(self, results: list[SearchResult]) -> list[SearchResult]:
        """按 (source, section_id) 补齐同章节兄弟 chunk，合并进 context（带总量上限）。

        知识库量级极小（~几十 chunk），直接 get_all() 内存分组即可，不改 Milvus schema
        （避免重建 collection 的数据迁移）。
        兄弟 chunk 未过 rerank、相关性未知，故：
        - 按与命中 chunk 的 chunk_index 邻近度排序补齐（离得越近语义越连续），避免大
          section 无差别灌入低相关段落稀释精确答案；
        - score 复用该 section 命中 chunk 的分数（不引入 score=0 破坏下游/SSE 语义），
          并打 is_expanded 标记供观测区分「真命中」与「章节补齐」。
        无 section_id 的旧数据天然退化不扩充；拉取失败降级保留精排结果，不阻断主链路。
        """
        hits = [
            (r.metadata["source"], r.metadata["section_id"],
             r.metadata.get("chunk_index") or 0, r.score)
            for r in results
            if r.metadata.get("source") and r.metadata.get("section_id")
        ]
        if not hits:
            return results
        try:
            all_docs = vector_store.get_all()
        except Exception as exc:
            logger.warning("event=rag_section_expand_fail error=%s", str(exc))
            return results

        # 按 (source, section_id) 分组兄弟 chunk，供就近补齐
        siblings: dict[tuple[str, str], list] = {}
        for d in all_docs:
            src, sid = d.metadata.get("source"), d.metadata.get("section_id")
            if src and sid:
                siblings.setdefault((src, sid), []).append(d)

        merged = list(results)
        seen = {r.id for r in results}
        for src, sid, hit_idx, hit_score in hits:
            near = sorted(
                siblings.get((src, sid), []),
                key=lambda d: abs((d.metadata.get("chunk_index") or 0) - hit_idx),
            )
            for d in near:
                if len(merged) >= _SECTION_EXPAND_TOTAL:
                    return merged
                if d.id in seen:
                    continue
                seen.add(d.id)
                meta = dict(d.metadata)
                meta["is_expanded"] = True
                merged.append(SearchResult(id=d.id, text=d.text, score=hit_score, metadata=meta))
        return merged


retriever = Retriever()
