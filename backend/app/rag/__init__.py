"""RAG 向量存储工厂。

MilvusVectorStore（LlamaIndex MilvusVectorStore 封装，独立 Milvus 服务）为唯一实现，
遵循 IVectorStore 接口，调用方（retriever/kb_store）无感。
"""
from app.rag.interfaces import IVectorStore
from app.rag.milvus_impl import MilvusVectorStore


vector_store: IVectorStore = MilvusVectorStore()
