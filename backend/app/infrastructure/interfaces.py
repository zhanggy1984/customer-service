"""资源层契约（Protocol，鸭子类型轻量）。

控制层（agent）只依赖本文件的接口类型 + app/infrastructure 门面导出，不直接
import 资源子模块具体实现。具体实现类无需显式继承 Protocol：结构匹配即满足，
mypy 在门面赋值处自动校验符合性。

取舍：turn_cache / metrics 是函数式模块（模块级函数 + 模块级状态），"可替换单元"
是整个模块而非对象，故不为其定义 Protocol；门面直接以模块对象导出（换实现 =
改门面绑定一行），与 services/__init__.py 的定位器模式同构。
"""
from collections.abc import AsyncIterator
from typing import Any, Protocol

from app.rag.interfaces import SearchResult


class ILLMGateway(Protocol):
    """LLM 网关（当前实现 DeepSeekGateway：Key 池化 + RPM + 排队 + 背压 + 熔断）。"""

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        thinking: bool | None = None,
    ) -> dict: ...

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[tuple[str, dict | None, str | None]]: ...

    async def init(self) -> None: ...
    async def close(self) -> None: ...


class ICooldown(Protocol):
    """共享冷却信号（Redis 广播，多节点语义）。"""

    async def is_open(self) -> bool: ...
    async def open(self) -> None: ...
    async def close(self) -> None: ...


class IMySQLPool(Protocol):
    """MySQL 异步连接池。"""

    async def init(self) -> None: ...
    async def close(self) -> None: ...
    async def fetchone(self, sql: str, params: tuple | list | None = None) -> dict | None: ...
    async def fetchall(self, sql: str, params: tuple | list | None = None) -> list[dict]: ...
    async def execute(self, sql: str, params: tuple | list | None = None) -> int: ...
    def transaction(self) -> Any: ...  # async context manager，协议仅作类型说明


class IRetriever(Protocol):
    """RAG 检索器（查询归一化 + L1 缓存 + 交叉编码重排 + 章节扩充）。"""

    async def search(self, query: str) -> list[SearchResult]: ...
    async def init(self) -> None: ...
    async def close(self) -> None: ...
    async def clear_cache(self) -> None: ...
