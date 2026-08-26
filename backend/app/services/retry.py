"""MySQL 写入重试装饰器 + DB 熔断。

对可重试的连接类错误（OperationalError / InterfaceError）指数退避重试
（0.1s / 0.2s / 0.4s），仍失败抛 ServiceUnavailableException 交由 Agent 兜底。

DB 熔断（挑战 3）：连续失败达阈值进入冷却，冷却期内所有 DB 调用（读写 6 个被
装饰方法）快速失败不再重试，防 MySQL 宕机时的重试风暴（每个请求串行打 4 次连接
可能压垮恢复中的 DB）。进程内存态，单实例可接受，进程重启即重置。
"""
import asyncio
import time
from functools import wraps

from asyncmy.errors import InterfaceError, OperationalError

from app.services.exceptions import ServiceUnavailableException
from app.utils.logger import logger

RETRYABLE_ERRORS = (OperationalError, InterfaceError)
BACKOFF_DELAYS = (0.1, 0.2, 0.4)  # 指数退避

DB_BREAKER_FAIL_THRESHOLD = 5  # 连续失败次数触发熔断
DB_BREAKER_COOLDOWN_SECONDS = 30  # 熔断冷却时长（秒），到期后半开放行一次尝试

_breaker = {"failures": 0, "open_until": 0.0}


def _breaker_open() -> bool:
    """熔断是否 open：冷却期内直接拒绝；冷却到期后重置计数放行（半开探测）。"""
    if time.time() < _breaker["open_until"]:
        return True
    if _breaker["failures"] >= DB_BREAKER_FAIL_THRESHOLD:
        _breaker["failures"] = 0  # 冷却到期：重置计数放行，靠下一次调用探测恢复
    return False


def _retry_on_db_error(fn):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        if _breaker_open():
            logger.warning("event=db_circuit_rejected", extra={"fn": fn.__name__})
            raise ServiceUnavailableException("数据库暂时不可用，请稍后重试")
        for attempt in range(len(BACKOFF_DELAYS) + 1):
            try:
                result = await fn(*args, **kwargs)
                _breaker["failures"] = 0  # 成功重置连续失败计数
                return result
            except RETRYABLE_ERRORS as exc:
                _breaker["failures"] += 1
                if _breaker["failures"] >= DB_BREAKER_FAIL_THRESHOLD:
                    _breaker["open_until"] = time.time() + DB_BREAKER_COOLDOWN_SECONDS
                    logger.error(
                        "event=db_circuit_open",
                        extra={"fn": fn.__name__, "cooldown": DB_BREAKER_COOLDOWN_SECONDS},
                    )
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
