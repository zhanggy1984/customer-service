"""DeepSeek Gateway：Key 池化 + RPM 追踪 + 排队 + 背压。

流程：
- 有 healthy Key → 执行；429 → mark_rate_limited 冷却后换 Key；5xx → 换 Key 重试（最多 2 次）
- 无 healthy Key → 排队等待（queue_timeout）；超时 → CapacityExceededError
- 全部 cooling → AllKeysDownError（上层触发规则引擎熔断）
"""
import asyncio

import httpx

from app.config import settings
from app.infrastructure.deepseek_keypool import KeyPool
from app.utils.logger import logger


class LLMUnavailableError(Exception):
    """LLM 调用失败（网络/超时/重试耗尽）。"""


class CapacityExceededError(Exception):
    """排队超时，容量不足。"""


class AllKeysDownError(Exception):
    """全部 Key 冷却，触发熔断降级。"""


class DeepSeekGateway:
    def __init__(self) -> None:
        keys = settings.deepseek_api_key_list
        if not keys:
            logger.error("event=deepseek_no_api_key 请在 .env 配置 DEEPSEEK_API_KEYS")
        self._pool = KeyPool(keys, settings.deepseek_per_key_rpm)
        self._client = httpx.AsyncClient(
            base_url=settings.deepseek_base_url,
            timeout=settings.deepseek_timeout_chat,
        )
        # 背压信号量：限制同时进行的 LLM 请求数（排队容量）
        self._semaphore = asyncio.Semaphore(settings.deepseek_queue_max_size)

    async def init(self) -> None:
        self._pool.start_cleanup()

    async def close(self) -> None:
        self._pool.stop_cleanup()
        await self._client.aclose()

    async def chat(self, messages: list[dict], model: str | None = None, timeout: float | None = None) -> dict:
        """兼容旧 DeepSeekClient.chat 签名。Key 池化 + 排队 + 背压。"""
        try:
            # 排队（背压）：并发超过容量时等待，超时抛容量不足
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=settings.deepseek_queue_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("event=queue_wait_timeout")
            raise CapacityExceededError("系统繁忙，请稍后再试") from None
        try:
            return await self._call(messages, model, timeout)
        finally:
            self._semaphore.release()

    async def _call(self, messages: list[dict], model: str | None, timeout: float | None) -> dict:
        payload = {
            "model": model or settings.deepseek_model_chat,
            "messages": messages,
            "stream": False,
        }
        request_timeout = timeout or settings.deepseek_timeout_chat

        for attempt in range(3):  # 换 Key 最多重试 2 次
            key = await self._pool.select_key()
            if key is None:
                if await self._pool.all_cooling():
                    raise AllKeysDownError("所有 DeepSeek Key 均不可用")
                raise CapacityExceededError("系统繁忙，请稍后再试")

            try:
                await key.record_request()
                resp = await self._client.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {key.api_key}"},
                    timeout=request_timeout,
                )
                logger.info(
                    "event=llm_call",
                    extra={"model": payload["model"], "key_index": key.index,
                           "key_rpm": await key.get_rpm(), "attempt": attempt, "status": resp.status_code},
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("retry-after", "30") or 30)
                    logger.warning("event=llm_429", extra={"key_index": key.index, "retry_after": retry_after})
                    await key.mark_rate_limited(retry_after)
                    continue  # 冷却后换 Key
                if resp.status_code >= 500:
                    logger.warning("event=llm_5xx", extra={"key_index": key.index, "status": resp.status_code})
                    continue  # 服务端错误换 Key
                resp.raise_for_status()  # 其他 4xx（参数错误等）不重试
            except httpx.TimeoutException:
                logger.error("event=llm_timeout", extra={"attempt": attempt})
                break  # 超时不重试
            except httpx.HTTPError as exc:
                logger.error("event=llm_http_error", extra={"attempt": attempt, "error": str(exc)})
                continue

        raise LLMUnavailableError("LLM 调用失败，请稍后重试")
