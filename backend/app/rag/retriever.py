"""RAG 检索器。

流程: query → embedding → 向量库检索 Top-10 → 交叉编码重排 Top-3 → score≥0.3 过滤。
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


retriever = Retriever()
