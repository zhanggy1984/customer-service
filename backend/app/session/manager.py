"""会话管理器（StorageRouter 提供 Redis/MySQL 自动切换）。

接口不变：create_session / get_session / update_session / close_session。
get_session 返回 None 表示会话不存在（上层自动重建）。
"""
import uuid

import redis.asyncio as aioredis

from app.config import settings
from app.session.models import Session
from app.session.storage_router import StorageRouter
from app.utils.logger import logger


class SessionManager:
    def __init__(self) -> None:
        self._router: StorageRouter | None = None

    async def init(self) -> None:
        if self._router is None:
            redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            self._router = StorageRouter(redis, settings.session_ttl)
            await self._router.start()

    async def close(self) -> None:
        if self._router is not None:
            await self._router.stop()
            await self._router.aclose_redis()
            self._router = None

    async def create_session(self, user_id: int) -> Session:
        session = Session(session_id=uuid.uuid4().hex, user_id=user_id)
        await self._router.save(session)
        logger.info("event=session_created", extra={"session_id": session.session_id, "user_id": user_id})
        return session

    async def get_session(self, sid: str) -> Session | None:
        return await self._router.load(sid)

    async def update_session(self, session: Session) -> None:
        session.trim(settings.session_max_messages)  # 消息体截断，防无限增长
        await self._router.save(session)

    async def close_session(self, sid: str) -> None:
        await self._router.delete(sid)
        logger.info("event=session_closed", extra={"session_id": sid})


session_manager = SessionManager()
