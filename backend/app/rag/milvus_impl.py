"""Milvus 向量存储实现（LlamaIndex MilvusVectorStore 封装）。

- 独立 Milvus 服务（docker-compose: milvus + etcd + minio），COSINE 度量。
- 实现 IVectorStore 接口（chroma 已移除，milvus 为唯一实现），调用方（retriever/kb_store）无感。
- 数据由 MySQL knowledge_docs 驱动重建（kb_store 一致性逻辑不变，换库零迁移）。
- 节点以 SOURCE 关系=source 入库（doc_id 字段），供 delete_by_source 按标量字段过滤。
- score 口径（P1.5 冒烟已校准）：Milvus COSINE 的 hit.distance 即余弦相似度（同向量=1.0），
  直接作相似度返回，与 Chroma 的 (1 - cosine_distance) 同口径，0.3 阈值语义一致。
"""
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.milvus import MilvusVectorStore as _LiMilvus

from app.config import settings
from app.rag.interfaces import Document, IVectorStore, SearchResult
from app.utils.logger import logger

# 全量对账/admin 拉取上限（小知识库量级足够；超大库需分页，非本场景）
_GET_ALL_LIMIT = 10000


class MilvusVectorStore(IVectorStore):
    def __init__(self) -> None:
        # 惰性建连：模块级 vector_store 实例化（import app.rag）时不连 Milvus，
        # 首次真实使用（count/add/search）才建立连接。chroma 移除后 milvus 是唯一实现，
        # eager 建连会让 import app.rag 在无 Milvus 环境（宿主机单测等）直接崩（pymilvus 构造即连）。
        self._store: _LiMilvus | None = None
        self._client = None
        self._collection = settings.milvus_collection

    @property
    def store(self) -> _LiMilvus:
        if self._store is None:
            self._store = _LiMilvus(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
                collection_name=settings.milvus_collection,
                dim=settings.embedding_dim,
                overwrite=False,  # 不重建已存在 collection（幂等连接）
                similarity_metric="COSINE",
            )
            self._client = self._store.client
            logger.info(
                "event=milvus_connected uri=%s collection=%s",
                settings.milvus_uri,
                settings.milvus_collection,
            )
        return self._store

    @property
    def client(self):
        self.store  # 触发建连
        return self._client

    def count(self) -> int:
        # stats row_count 在新建 collection 插入后、flush 前会滞后为 0（P1.5 冒烟发现），
        # 用 count(*) 强一致读即时准确，保证插入后 search 的 count()<=0 门卫不误伤。
        res = self.client.query(
            self._collection,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(res[0].get("count(*)", 0))

    async def add_documents(self, docs: list[Document], embeddings: list[list[float]] | None = None) -> None:
        if not docs:
            return
        if embeddings is None:
            # LlamaIndex MilvusVectorStore 未配置 embed_model，无法自行嵌入
            raise ValueError("MilvusVectorStore 需要显式传入 embeddings")
        nodes = []
        for i, d in enumerate(docs):
            meta = d.metadata or {}
            n = TextNode(id_=d.id, text=d.text, metadata=meta, embedding=embeddings[i])
            # TextNode 的 ref_doc_id 由 relationships[SOURCE] 派生（构造函数传 ref_doc_id 会被忽略，
            # P1.5 冒烟发现 doc_id 全为 "None"，delete_by_source 失效）。source 作 doc_id 供标量过滤。
            n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=meta.get("source", ""))
            nodes.append(n)
        self.store.add(nodes)
        logger.info("event=rag_add_docs store=milvus count=%s", len(docs))

    async def search(self, query_embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        if self.count() <= 0:
            return []
        # Milvus store.query 需要 VectorStoreQuery（QueryBundle 无 .mode 会报错，P1.5 冒烟发现）。
        # COSINE 度量下 hit.distance 即余弦相似度本身（同向量=1.0，实测），越高越相似，
        # 与 Chroma 的 (1 - cosine_distance) 同口径；0.3 阈值与 rerank 降级排序均按"高=相似"。
        qb = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        res = self.store.query(qb)
        return [
            SearchResult(
                id=n.id_,
                text=n.text,
                score=float(s),
                metadata=n.metadata or {},
            )
            for n, s in zip(res.nodes, res.similarities)
        ]

    async def delete(self, ids: list[str]) -> None:
        if ids:
            self.client.delete(self._collection, ids=ids)
            logger.info("event=rag_delete_docs store=milvus count=%s", len(ids))

    async def delete_by_source(self, source: str) -> None:
        res = self.client.query(
            self._collection, filter=f'doc_id == "{source}"', output_fields=["id"], consistency_level="Strong"
        )
        ids = [r["id"] for r in res]
        if ids:
            self.client.delete(self._collection, ids=ids)
        logger.info("event=rag_delete_by_source store=milvus source=%s removed=%s", source, len(ids))

    def count_by_source(self, source: str) -> int:
        res = self.client.query(
            self._collection, filter=f'doc_id == "{source}"', output_fields=["id"], consistency_level="Strong"
        )
        return len(res)

    def get_all(self) -> list[Document]:
        """全量拉取（admin 全量对账的孤儿清理用）。metadata 以 JSON 字段存储，需还原 dict。"""
        import json

        res = self.client.query(
            self._collection,
            output_fields=["id", "text", "_node_content"],
            limit=_GET_ALL_LIMIT,
            consistency_level="Strong",
        )
        docs = []
        for row in res:
            # metadata 无独立字段，存在 _node_content 的节点 JSON 里（LlamaIndex 序列化）
            meta = {}
            raw = row.get("_node_content") or ""
            if isinstance(raw, str):
                try:
                    meta = json.loads(raw).get("metadata") or {}
                except (ValueError, TypeError):
                    meta = {}
            docs.append(Document(id=row.get("id", ""), text=row.get("text", ""), metadata=meta))
        return docs
