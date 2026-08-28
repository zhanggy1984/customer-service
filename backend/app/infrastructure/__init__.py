"""资源层统一门面（对齐 services/__init__.py 的定位器模式）。

规则：控制层（agent）只允许从本门面取资源，禁止直接 import 资源子模块具体实现。
- 实例资源：导出「语义化名字: 接口类型 = 单例」，换实现只改本文件一处；
- 函数式模块（turn_cache/metrics）：以模块对象导出（可替换单元 = 模块）；
- load-bearing：保持与 deepseek.py / mysql.py / retriever.py 的模块级单例同一对象
  （勿重新实例化——测试按对象身份 patch 依赖它，见 test_infrastructure_facade.py）；
- retriever / RetrievalUnavailableError 经 __getattr__ 惰性导出：rag/retriever.py 顶层
  import 本门面（turn_cache），若门面顶层同步 import retriever，则 retriever 作为导入链
  上游时（如单独跑 tests/test_rag.py）门面会从 partial 初始化的 retriever 取符号而
  ImportError——惰性导出让 retriever 在门面初始化完成后才加载，破除该环。
"""
from typing import Any

from app.infrastructure import metrics
from app.infrastructure import turn_cache
from app.infrastructure.cooldown import RedisCooldown
from app.infrastructure.deepseek import LLM_FALLBACK_ERRORS, deepseek_client
from app.infrastructure.interfaces import ICooldown, ILLMGateway, IMySQLPool, IRetriever
from app.infrastructure.mysql import mysql_pool as _mysql_pool
from app.infrastructure.turn_cache import normalize_query
from app.rag.interfaces import source_label

# 实例资源：语义化名字 : 接口类型 = 单例（结构匹配由 Protocol 在赋值处校验）
llm_gateway: ILLMGateway = deepseek_client
mysql_pool: IMySQLPool = _mysql_pool
# retriever 仅声明类型（惰性加载，见 __getattr__）：保留静态检查且不触发 rag 导入
retriever: IRetriever


def __getattr__(name: str) -> Any:
    """惰性加载 app.rag.retriever 符号，破除 infrastructure → rag → infrastructure 导入环。"""
    if name == "retriever":
        from app.rag.retriever import retriever as _r
        return _r
    if name == "RetrievalUnavailableError":
        from app.rag.retriever import RetrievalUnavailableError as _err
        return _err
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RedisCooldown",
    "LLM_FALLBACK_ERRORS",
    "normalize_query",
    "source_label",
    "RetrievalUnavailableError",
    "ICooldown",
    "ILLMGateway",
    "IMySQLPool",
    "IRetriever",
    "metrics",
    "turn_cache",
    "llm_gateway",
    "mysql_pool",
    "retriever",
]
