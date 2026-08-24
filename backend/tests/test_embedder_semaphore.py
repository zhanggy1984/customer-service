"""embedder 串行化单测：并发 embed 最多一个同时进入 _encode。

根治首次并发加载 SentenceTransformer 的 meta-tensor 竞态（5 并发 3 失败）。
信号量保证同一时刻仅一个调用在真实 encode，验证串行化生效。
"""
import asyncio
import time

import pytest

from app.rag.embedder import Embedder


@pytest.mark.asyncio
async def test_embed_documents_serialized(monkeypatch):
    e = Embedder()
    stats = {"active": 0, "max": 0, "calls": 0}

    def fake_encode(texts):
        stats["active"] += 1
        stats["max"] = max(stats["max"], stats["active"])
        stats["calls"] += 1
        time.sleep(0.05)  # 拉大重叠窗口，验证信号量确实互斥
        stats["active"] -= 1
        return [[0.1] * e.dim for _ in texts]

    monkeypatch.setattr(e, "_encode", fake_encode)
    await asyncio.gather(*[e.embed_documents(["a", "b"]) for _ in range(5)])
    assert stats["calls"] == 5
    assert stats["max"] == 1  # 全程串行，无并发进入 _encode


@pytest.mark.asyncio
async def test_embed_query_uses_documents(monkeypatch):
    """embed_query 复用 embed_documents 路径（同一信号量串行）。"""
    e = Embedder()
    monkeypatch.setattr(e, "_encode", lambda texts: [[0.5] * e.dim] * len(texts))
    vec = await e.embed_query("退货政策")
    assert len(vec) == e.dim
