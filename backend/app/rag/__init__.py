"""RAG 向量存储工厂。

VECTOR_STORE 环境变量决定实现：
- chroma → ChromaVectorStore（当前）
- milvus → 未来接入 Milvus（实现同一 IVectorStore 接口）
"""
from app.rag.chroma_impl import ChromaVectorStore
from app.rag.interfaces import IVectorStore

vector_store: IVectorStore = ChromaVectorStore()
