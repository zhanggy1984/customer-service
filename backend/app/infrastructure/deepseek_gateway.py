"""DeepSeek Gateway：Key 池化 + RPM 追踪 + 排队 + 背压。

流程：
- 有 healthy Key → 执行；429 → mark_rate_limited 冷却后换 Key；5xx → 换 Key 重试（最多 2 次）
- 无 healthy Key → 排队等待（queue_timeout）；超时 → CapacityExceededError
- 全部 cooling → AllKeysDownError（上层触发规则引擎熔断）
"""
import asyncio
import json

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

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict:
        """兼容旧 DeepSeekClient.chat 签名。Key 池化 + 排队 + 背压。

        temperature 可选：不传（None）不加该字段，走 DeepSeek 默认采样；
        意图分类等确定性场景传低温度（如 0.1）。
        tools/tool_choice 可选（工具决策循环用）：不传（None）则不加
        tools 字段，OpenAI 兼容格式原样透传给 DeepSeek，现有调用方无感。
        """
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
            return await self._call(messages, model, timeout, temperature, tools, tool_choice)
        finally:
            self._semaphore.release()

    async def _call(
        self,
        messages: list[dict],
        model: str | None,
        timeout: float | None,
        temperature: float | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict:
        payload = {
            "model": model or settings.deepseek_model_chat,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
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

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ):
        """流式 chat（契约：token 级流式透出）。保留 Key 池化/排队/背压语义。

        yield (delta_content, usage_or_None)：delta 非空为内容增量；
        usage 仅在最后一个 chunk（include_usage）出现，通常为 None。

        重试边界（决策：首个 content delta 前可换 Key，之后绝不重试）：
        - HTTP 层 429/5xx → 换 Key 重试（未进内容流，安全）
        - 流内首个 content delta 之前的异常 → 换 Key 重试
        - 首个 content delta 之后的任何异常 → 直接上抛，不回滚已流出 token
        """
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=settings.deepseek_queue_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("event=queue_wait_timeout")
            raise CapacityExceededError("系统繁忙，请稍后再试") from None
        try:
            async for item in self._stream(messages, model, timeout, temperature):
                yield item
        finally:
            self._semaphore.release()

    async def _stream(
        self,
        messages: list[dict],
        model: str | None,
        timeout: float | None,
        temperature: float | None = None,
    ):
        payload = {
            "model": model or settings.deepseek_model_chat,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},  # 最后一个 chunk 携带 usage
        }
        if temperature is not None:
            payload["temperature"] = temperature
        request_timeout = timeout or settings.deepseek_timeout_chat

        for attempt in range(3):  # 换 Key 最多重试 2 次
            key = await self._pool.select_key()
            if key is None:
                if await self._pool.all_cooling():
                    raise AllKeysDownError("所有 DeepSeek Key 均不可用")
                raise CapacityExceededError("系统繁忙，请稍后再试")

            started = False  # 是否已流出首个 content delta
            try:
                await key.record_request()
                async with self._client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {key.api_key}"},
                    timeout=request_timeout,
                ) as resp:
                    if resp.status_code != 200:
                        # HTTP 层失败：未进内容流，安全换 Key 重试
                        if resp.status_code == 429:
                            retry_after = int(resp.headers.get("retry-after", "30") or 30)
                            logger.warning(
                                "event=llm_429", extra={"key_index": key.index, "retry_after": retry_after}
                            )
                            await key.mark_rate_limited(retry_after)
                            continue
                        if resp.status_code >= 500:
                            logger.warning(
                                "event=llm_5xx", extra={"key_index": key.index, "status": resp.status_code}
                            )
                            continue
                        resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        # usage 在最后一个 chunk（include_usage），choices 为空
                        if chunk.get("usage"):
                            u = chunk["usage"]
                            # 7.4 透传 cache 字段（评测平台按 cache_hit_price 计命中成本）
                            yield "", {
                                "prompt_tokens": u.get("prompt_tokens", 0),
                                "completion_tokens": u.get("completion_tokens", 0),
                                "total_tokens": u.get("total_tokens", 0),
                                "prompt_cache_hit_tokens": u.get("prompt_cache_hit_tokens", 0),
                                "prompt_cache_miss_tokens": u.get("prompt_cache_miss_tokens", 0),
                            }
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content") or ""
                        if content:
                            started = True
                            yield content, None
                    return  # 正常流结束
            except httpx.TimeoutException:
                logger.error(
                    "event=llm_stream_timeout", extra={"attempt": attempt, "started": started}
                )
                if started:
                    raise LLMUnavailableError("LLM 流式中断") from None
                continue  # 未流出内容，可换 Key 重试
            except httpx.HTTPError as exc:
                logger.error(
                    "event=llm_stream_http_error",
                    extra={"attempt": attempt, "started": started, "error": str(exc)},
                )
                if started:
                    raise LLMUnavailableError("LLM 流式中断") from None
                continue

        raise LLMUnavailableError("LLM 调用失败，请稍后重试")
