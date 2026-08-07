"""知识库启动初始化（入口转发到 kb_store，避免与 ChromaDB 直接耦合）。

职责（见 kb_store.ensure_init）：
1. MySQL 表空 → 从 knowledge/*.md 灌入文档源数据；
2. 从 MySQL 同步 ChromaDB（仅增量补偿 pending 行；全量重建只在
   ChromaDB 数据丢失 / 首次灌入时执行，不成为常态路径）。
"""
from app.utils.logger import logger
from app.rag.kb_store import ensure_init


async def init_knowledge() -> None:
    try:
        await ensure_init()
    except Exception as exc:
        logger.error("event=rag_init_failed error=%s", str(exc))
