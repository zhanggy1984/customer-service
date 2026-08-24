"""Embedding 封装：BAAI/bge-small-zh-v1.5（512 维）。

- 模型懒加载（首次调用下载，之后缓存），下载走 HF 镜像（国内可拉）。
- encode 是 CPU 阻塞操作，放入线程池避免阻塞事件循环。
- 并发安全：首次并发调用会同时触发 SentenceTransformer 加载，PyTorch
  meta-device 迁移阶段多线程 encode 会抛 "Cannot copy out of meta tensor"
  （实测 5 并发 3 失败）。用信号量将全部 embed 串行化，首请求单线程加载，
  之后复用已加载模型，根治竞态。embed 单次 ~50ms，串行化对整体延迟影响可忽略。
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
    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(1)

    @property
    def dim(self) -> int:
        return settings.embedding_dim

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = _load_model()
        vecs = model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 串行化：模型加载/推理均为单线程安全，避免并发首次加载竞态
        async with self._sem:
            return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


embedder = Embedder()
