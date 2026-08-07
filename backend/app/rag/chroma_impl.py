"""ChromaDB 向量存储实现（cosine 距离）。

两种部署形态（config.chroma_host 决定）：
- chroma_host 为空 → PersistentClient 嵌入式（开发单机）
- chroma_host 非空 → HttpClient 连接独立 chroma 服务（生产/多实例）
"""
import chromadb
from chromadb.config import Settings

from app.config import settings
from app.rag.interfaces import Document, IVectorStore, SearchResult
from app.utils.logger import logger


class ChromaVectorStore(IVectorStore):
    def __init__(self, persist_dir: str | None = None) -> None:
        cfg = Settings(anonymized_telemetry=False)
        if settings.chroma_host:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=cfg,
            )
            logger.info("event=chroma_mode_http host=%s port=%s", settings.chroma_host, settings.chroma_port)
        else:
            path = persist_dir or settings.chroma_persist_dir
            self._client = chromadb.PersistentClient(path=path, settings=cfg)
        self._collection = self._client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    async def add_documents(self, docs: list[Document], embeddings: list[list[float]] | None = None) -> None:
        if not docs:
            return
        ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        metadatas = [d.metadata or {} for d in docs]
        if embeddings:
            self._collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        else:
            self._collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logger.info("event=rag_add_docs count=%s", len(docs))

    async def search(self, query_embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        total = self._collection.count()
        if total <= 0:
            return []
        n = min(top_k, total)
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0]
        texts = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        results = []
        for i in range(len(ids)):
            # cosine 距离 → 相似度
            score = 1.0 - distances[i]
            results.append(
                SearchResult(id=ids[i], text=texts[i], score=score, metadata=metas[i] or {})
            )
        return results

    async def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)
            logger.info("event=rag_delete_docs count=%s", len(ids))

    async def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})
        logger.info("event=rag_delete_by_source source=%s", source)

    def count_by_source(self, source: str) -> int:
        # chroma 1.4 的 count() 不支持 where，改用 get(where=...) 数 ids
        res = self._collection.get(where={"source": source})
        return len(res.get("ids", []))

    def get_all(self) -> list[Document]:
        res = self._collection.get(include=["documents", "metadatas"])
        ids = res.get("ids", [])
        texts = res.get("documents", [])
        metas = res.get("metadatas", [])
        return [
            Document(id=ids[i], text=texts[i], metadata=metas[i] or {})
            for i in range(len(ids))
        ]
