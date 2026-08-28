"""资源层门面身份与表面测试。

load-bearing：门面必须 re-export 既有模块级单例**同一对象**，禁止重新实例化——
各测试按对象身份 patch 单例（`monkeypatch.setattr(x.llm_gateway, "chat", ...)`），
若门面另建实例（如 `llm_gateway = DeepSeekGateway()`），patch 的旧单例不再是 agent
实际使用的对象，测试将静默失效。本文件即防该回归的锚点。
"""
from app.infrastructure import (
    ICooldown,
    ILLMGateway,
    IMySQLPool,
    IRetriever,
    llm_gateway,
    metrics,
    mysql_pool,
    normalize_query,
    retriever,
    source_label,
    turn_cache,
)
from app.infrastructure.deepseek import deepseek_client
from app.infrastructure.mysql import mysql_pool as _mysql_pool
from app.rag.retriever import retriever as _retriever


def test_facade_preserves_singleton_identity():
    """门面导出 == 源单例同一对象（勿重新实例化，身份 patch 依赖它）。"""
    assert llm_gateway is deepseek_client
    assert mysql_pool is _mysql_pool
    assert retriever is _retriever


def test_facade_surface():
    """门面导出面完整：LLM 网关方法 + 函数式模块 + 纯函数 + 接口类型。"""
    for m in ("chat", "chat_stream", "init", "close"):
        assert callable(getattr(llm_gateway, m))
    assert callable(turn_cache.normalize_query)
    assert callable(metrics.inc)
    assert callable(normalize_query)
    assert callable(source_label)
    for iface in (ILLMGateway, IMySQLPool, IRetriever, ICooldown):
        assert isinstance(iface, type)
