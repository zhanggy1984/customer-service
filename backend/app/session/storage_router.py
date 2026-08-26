"""StorageRouter：会话存储 Redis 主 / MySQL 兜底，自动切换。

- 写入：Redis 同步（失败切 mysql_fallback）+ MySQL 异步双写
- 读取：Redis 优先；Redis 不可用或未命中 → MySQL
- 恢复：后台每 5s ping Redis，恢复后自动切回（event=storage_mode_switch 日志）
"""
import asyncio
import json

from app.infrastructure.mysql import mysql_pool
from app.session.models import Message, Session
from app.utils.logger import logger


class StorageRouter:
    def __init__(self, redis, ttl: int) -> None:
        self._redis = redis
        self._ttl = ttl
        self._mode = "redis"  # redis | mysql_fallback
        self._monitor_task: asyncio.Task | None = None

    @staticmethod
    def _key(sid: str) -> str:
        return f"session:{sid}"

    async def start(self) -> None:
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def aclose_redis(self) -> None:
        await self._redis.aclose()

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(5)
            if self._mode == "mysql_fallback":
                try:
                    await self._redis.ping()
                    self._mode = "redis"
                    logger.info("event=storage_mode_switch", extra={"mode": "redis"})
                except Exception:
                    pass

    def _fallback(self, exc: Exception) -> None:
        if self._mode != "mysql_fallback":
            self._mode = "mysql_fallback"
            logger.warning("event=storage_mode_switch", extra={"mode": "mysql_fallback", "error": str(exc)})

    async def save(self, session: Session) -> None:
        payload = session.model_dump_json()
        try:
            await self._redis.set(self._key(session.session_id), payload, ex=self._ttl)
        except Exception as exc:
            self._fallback(exc)
        # MySQL 异步双写（不阻塞主流程）
        await self._save_mysql(session)

    async def load(self, sid: str) -> Session | None:
        if self._mode == "redis":
            try:
                raw = await self._redis.get(self._key(sid))
                if raw:
                    await self._redis.expire(self._key(sid), self._ttl)  # 滑动过期
                    return Session.model_validate_json(raw)
            except Exception as exc:
                self._fallback(exc)
        return await self._load_mysql(sid)

    async def delete(self, sid: str) -> None:
        try:
            await self._redis.delete(self._key(sid))
        except Exception:
            pass
        await mysql_pool.execute("DELETE FROM conversation_history WHERE session_id=%s", (sid,))
        # 级联回收护栏判定日志（既有缺口：原实现只删会话，tool_call_log 残留无限累积）
        await mysql_pool.execute("DELETE FROM tool_call_log WHERE session_id=%s", (sid,))

    async def is_expired(self, sid: str, days: int) -> bool:
        """会话最后活跃是否超保留期（仅 MySQL 兜底恢复场景）。

        Redis 命中=1h 内活跃必然未超期，这里只为 Redis miss 走 _load_mysql 的会话判断。
        判据落在 MySQL 侧：NOW() 与 created_at 的 CURRENT_TIMESTAMP 同会话时区基准，
        Python 不生成 cutoff，避免 aware/naive 混比及时区错位。
        """
        row = await mysql_pool.fetchone(
            "SELECT created_at FROM conversation_history WHERE session_id=%s "
            "AND created_at < NOW() - INTERVAL %s DAY ORDER BY id DESC LIMIT 1",
            (sid, days),
        )
        return row is not None

    # ---------- MySQL 存取 ----------

    async def _save_mysql(self, session: Session) -> None:
        try:
            # agent_state 可能含 datetime（状态机还原订单 delivered_at），
            # 不加 default 会导致 json.dumps 抛 TypeError，MySQL 会话兜底静默失效
            agent_state_payload = json.dumps(
                {"agent_state": session.agent_state, "snapshots": session.snapshots},
                ensure_ascii=False,
                default=str,
            )
            await mysql_pool.execute(
                "DELETE FROM conversation_history WHERE session_id=%s", (session.session_id,)
            )
            await mysql_pool.execute(
                "INSERT INTO conversation_history (session_id, user_id, intent, messages, agent_state, summary) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    session.session_id,
                    session.user_id,
                    session.intent,
                    # model_dump() 默认保留 datetime 对象，裸 json.dumps 会抛 TypeError，
                    # 与 Redis 的 model_dump_json() 口径一致，mode="json" 输出 ISO 字符串
                    json.dumps([m.model_dump(mode="json") for m in session.messages], ensure_ascii=False),
                    agent_state_payload,
                    "",
                ),
            )
        except Exception as exc:
            logger.error("event=session_mysql_save_error", extra={"session_id": session.session_id, "error": str(exc)})

    async def _load_mysql(self, sid: str) -> Session | None:
        try:
            row = await mysql_pool.fetchone(
                "SELECT user_id, intent, messages, agent_state FROM conversation_history "
                "WHERE session_id=%s ORDER BY id DESC LIMIT 1",
                (sid,),
            )
        except Exception:
            return None
        if not row:
            return None
        try:
            messages = [Message(**m) for m in json.loads(row["messages"] or "[]")]
            st = json.loads(row["agent_state"] or "{}")
        except (json.JSONDecodeError, TypeError):
            messages, st = [], {}
        return Session(
            session_id=sid,
            user_id=row["user_id"],
            intent=row["intent"],
            messages=messages,
            agent_state=st.get("agent_state"),
            snapshots=st.get("snapshots", {}),
        )
