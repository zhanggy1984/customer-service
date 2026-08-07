"""Embedding 封装：BAAI/bge-small-zh-v1.5（512 维）。

- 模型懒加载（首次调用下载，之后缓存），下载走 HF 镜像（国内可拉）。
- encode 是 CPU 阻塞操作，放入线程池避免阻塞事件循环。
- 未来可切换 DeepSeek Embedding API（复用 Gateway Key 池），改这里实现即可。
"""
import asyncio
import os
from functools import lru_cache

from app.config import settings
from app.utils.logger import logger

# 国内下载 HuggingFace 模型走镜像；用户可在环境变量中覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    logger.info("event=embedder_load model=%s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


class Embedder:
    @property
    def dim(self) -> int:
        return settings.embedding_dim

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = _load_model()
        vecs = model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


embedder = Embedder()
