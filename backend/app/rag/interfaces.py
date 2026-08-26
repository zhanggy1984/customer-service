"""向量存储抽象接口。

当前实现: MilvusVectorStore（LlamaIndex MilvusVectorStore 封装，独立 Milvus 服务）。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def source_label(metadata: dict) -> str:
    """溯源标签：source 拼 heading_path（如 "return_policy > 退货时限"）。

    chunk 带标题路径时提示来自哪篇文档哪个章节；无标题数据退化为纯 source。
    """
    src = metadata.get("source", "")
    path = metadata.get("heading_path") or []
    return " > ".join([src] + list(path)) if path else src


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    text: str
    score: float  # 0.0-1.0，越高越相关
    metadata: dict = field(default_factory=dict)


class IVectorStore(ABC):
    @abstractmethod
    async def add_documents(self, docs: list[Document], embeddings: list[list[float]] | None = None) -> None:
        """写入文档。embeddings 为空则由存储侧自行嵌入。"""

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        """按查询向量检索，按相似度降序返回 top_k 条。"""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """按文档 id 删除。"""

    @abstractmethod
    async def delete_by_source(self, source: str) -> None:
        """按元数据 source（文档标题）删除全部 chunks。

        知识库以文档为管理单位，重建一篇文档 = delete_by_source + add。
        """

    @abstractmethod
    def count(self) -> int:
        """集合内文档总数。"""

    @abstractmethod
    def count_by_source(self, source: str) -> int:
        """指定文档（source）的 chunk 数。"""
