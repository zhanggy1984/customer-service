"""MySQL 写入重试装饰器。

对可重试的连接类错误（OperationalError / InterfaceError）指数退避重试
（0.1s / 0.2s / 0.4s），仍失败抛 ServiceUnavailableException 交由 Agent 兜底。
"""
import asyncio
from functools import wraps

from asyncmy.errors import InterfaceError, OperationalError

from app.services.exceptions import ServiceUnavailableException
from app.utils.logger import logger

RETRYABLE_ERRORS = (OperationalError, InterfaceError)
BACKOFF_DELAYS = (0.1, 0.2, 0.4)  # 指数退避


def _retry_on_db_error(fn):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        for attempt in range(len(BACKOFF_DELAYS) + 1):
            try:
                return await fn(*args, **kwargs)
            except RETRYABLE_ERRORS as exc:
                if attempt == len(BACKOFF_DELAYS):
                    logger.error(
                        "event=db_write_failed",
                        extra={"fn": fn.__name__, "error": str(exc)},
                    )
                    raise ServiceUnavailableException("数据库暂时不可用，请稍后重试") from exc
                delay = BACKOFF_DELAYS[attempt]
                logger.warning(
                    "event=db_write_retry",
                    extra={"fn": fn.__name__, "attempt": attempt + 1, "delay": delay, "error": str(exc)},
                )
                await asyncio.sleep(delay)
    return wrapper
