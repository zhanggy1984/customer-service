"""Embedding 封装：BAAI/bge-small-zh-v1.5（512 维），底层走 LlamaIndex HuggingFaceEmbedding。

- 模型懒加载（首次调用下载，之后缓存），下载走 HF 镜像（国内可拉）。
- encode 是 CPU 阻塞操作，放入线程池避免阻塞事件循环。
- 并发安全：首次并发调用会同时触发模型加载，PyTorch meta-device 迁移阶段
  多线程 encode 会抛 "Cannot copy out of meta tensor"。用信号量将全部 embed
  串行化，首请求单线程加载，之后复用已加载模型，根治竞态。embed 单次 ~50ms，
  串行化对整体延迟影响可忽略。
- 输出与现状 SentenceTransformer.encode(normalize_embeddings=True) 对齐：
  get_text_embedding_batch 取原始向量后手动 L2 归一化（幂等，双保险）。
- 未来可切换 DeepSeek Embedding API（复用 Gateway Key 池），改这里实现即可。
"""
import asyncio
import math
import os
from functools import lru_cache

from app.config import settings
from app.utils.logger import logger

# 国内下载 HuggingFace 模型走镜像；用户可在环境变量中覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


@lru_cache(maxsize=1)
def _load_model():
    from huggingface_hub.constants import HF_HUB_CACHE
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    logger.info("event=embedder_load model=%s", settings.embedding_model)
    # llama-index-core 的 get_cache_dir() 默认指向 ~/.cache/llama_index，
    # 与 HF 标准缓存 ~/.cache/huggingface/hub 不一致：容器挂载的 hf_cache volume
    # 里预置的模型它找不到，干净环境会联网拉取（本机连 hf-mirror 必挂）。
    # 显式指到 HF 缓存目录，offline 读本地缓存即可加载。
    return HuggingFaceEmbedding(
        model_name=settings.embedding_model, cache_folder=HF_HUB_CACHE
    )


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1e-12
    return [v / norm for v in vec]


class Embedder:
    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(1)

    @property
    def dim(self) -> int:
        return settings.embedding_dim

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = _load_model()
        vecs = model.get_text_embedding_batch(texts, show_progress=False)
        # L2 归一化：与 bge 常规用法对齐，保证余弦检索口径稳定
        return [_l2_normalize(v) for v in vecs]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 串行化：模型加载/推理均为单线程安全，避免并发首次加载竞态
        async with self._sem:
            return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


embedder = Embedder()
