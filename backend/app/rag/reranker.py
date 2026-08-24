"""交叉编码重排器：bge-reranker-base（真重排，修复"按相似度排序冒充重排"）。

- 对 (query, chunk) 交叉编码打分，比仅靠向量余弦更准。
- 只调整顺序，保留原始 cosine score 字段（阈值过滤仍基于 cosine 0.3，口径不变）。
- 模型约 1GB，懒加载；CPU 阻塞，放入线程池；信号量串行化防并发加载竞态。
- 重排失败由 retriever 兜底降级为按相似度排序，不阻断检索。
"""
import asyncio
from functools import lru_cache

from app.rag.interfaces import SearchResult
from app.utils.logger import logger

RE_RANK_K = 3


@lru_cache(maxsize=1)
def _load_reranker():
    from llama_index.core.postprocessor import SentenceTransformerRerank

    from app.config import settings

    logger.info("event=reranker_load model=%s", settings.rerank_model)
    return SentenceTransformerRerank(model=settings.rerank_model, top_n=RE_RANK_K)


class Reranker:
    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(1)

    def _rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if len(results) <= 1:
            return results
        from llama_index.core.schema import NodeWithScore, TextNode

        nodes = [
            NodeWithScore(node=TextNode(id_=r.id, text=r.text, metadata=r.metadata), score=r.score)
            for r in results
        ]
        ranked = _load_reranker().postprocess_nodes(nodes, query_str=query)
        by_id = {r.id: r for r in results}
        return [by_id[n.node.id_] for n in ranked if n.node.id_ in by_id]

    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        # 串行化：与 embedder 同理，防交叉编码器并发加载竞态
        async with self._sem:
            return await asyncio.to_thread(self._rerank, query, results)


reranker = Reranker()
