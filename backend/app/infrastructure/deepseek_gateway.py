"""DeepSeek Gateway：Key 池化 + RPM 追踪 + 排队 + 背压。

流程：
- 有 healthy Key → 执行；429 → mark_rate_limited 冷却后换 Key；5xx → 换 Key 重试（最多 2 次）
- 无 healthy Key → 排队等待（queue_timeout）；超时 → CapacityExceededError
- 全部 cooling → AllKeysDownError（上层触发规则引擎熔断）
"""
import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from app.config import settings
from app.infrastructure import metrics
from app.infrastructure.cooldown import RedisCooldown
from app.infrastructure.deepseek_keypool import KeyPool
from app.utils.logger import logger


class LLMUnavailableError(Exception):
    """LLM 调用失败（网络/超时/重试耗尽）。"""


class StreamInterruptedError(LLMUnavailableError):
    """流式生成中途（已流出首个 content delta）后的连接中断。

    区别于 LLMUnavailableError：这是单次请求的连接级抖动/用户断连，不代表网关整体
    故障，不参与熔断累计（挑战1：5 个长流中途断一下不应误熔断全网关）。
    仍被上层 LLM_FALLBACK_ERRORS 捕获（子类），走规则引擎兜底，行为不变。
    """


class CapacityExceededError(Exception):
    """排队超时，容量不足。"""


class AllKeysDownError(Exception):
    """全部 Key 冷却，触发熔断降级。"""


# LLM 网关熔断（仿 services/retry.py 的 DB _breaker 模式）：累计"一次逻辑调用彻底失败"
# （换 Key 重试耗尽 / 超时 break），连续达阈值进入冷却，冷却期内入口直接拒绝零网络尝试。
# 429 全冷（AllKeysDown）与本地排队超时（CapacityExceeded）不走此熔断——前者已由 KeyPool
# 冷却机制覆盖，后者是本进程负载非上游持续故障。进程内存态，单实例可接受，进程重启即重置。
# 阈值取 2 而非更高（挑战2）：计入熔断的都是强故障信号——超时类失败是等满 timeout 才失败
# （慢挂确定性信号，每失败代价 ≈1×timeout），若阈值 5，慢挂需 5×timeout 才熔断，期间每请求
# 都白挂满超时；取 2 则 2×timeout 即进入冷却，用户快速转入规则引擎兜底。
_LLM_BREAKER_FAIL_THRESHOLD = 2  # 连续失败次数触发熔断
_LLM_BREAKER_COOLDOWN = 30.0  # 熔断冷却时长（秒），到期后半开放行一次尝试
_LLM_RETRY_BACKOFF = (0.1, 0.2)  # 换 Key 重试退避（指数，两档对应 ≤2 次重试）


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
        # LLM 熔断状态（实例属性：测试可注入/天然隔离，生产单例等效全局）。
        # 计数留本地，冷却期经 RedisCooldown 广播全局（多节点共享熔断信号）
        self._breaker = {"failures": 0, "open_until": 0.0}
        self._cooldown = RedisCooldown("llm", _LLM_BREAKER_COOLDOWN)

    async def _breaker_open(self) -> bool:
        """熔断是否 open：本地冷却期内，或任一节点已广播熔断 → 直接拒绝。"""
        if time.time() < self._breaker["open_until"]:
            return True
        if await self._cooldown.is_open():  # 共享信号：他节点已熔断 → 本节点提前降级
            return True
        if self._breaker["failures"] >= _LLM_BREAKER_FAIL_THRESHOLD:
            self._breaker["failures"] = 0  # 冷却到期：重置计数放行，靠下一次调用探测恢复
        return False

    async def _breaker_fail(self) -> None:
        """记录一次熔断失败；达阈值进入冷却并广播全局。"""
        self._breaker["failures"] += 1
        if self._breaker["failures"] >= _LLM_BREAKER_FAIL_THRESHOLD:
            self._breaker["open_until"] = time.time() + _LLM_BREAKER_COOLDOWN
            await self._cooldown.open()  # 广播：让其他节点不必再打失败的 LLM
            metrics.inc("llm_circuit_open")
            logger.error(
                "event=llm_circuit_open",
                extra={"cooldown": _LLM_BREAKER_COOLDOWN},
            )

    async def _breaker_reset(self) -> None:
        """调用成功：重置连续失败计数；仅本地广播方成功时清除共享熔断信号。

        他节点成功不 DEL（防撤销他人广播的共享降级信号，多节点语义）。
        """
        self._breaker["failures"] = 0
        if self._breaker["open_until"] > time.time():
            await self._cooldown.close()

    async def _backoff_sleep(self, attempt: int) -> None:
        """换 Key 重试前的指数退避；最后一次尝试失败不再等（循环将退出，白睡无意义）。"""
        if attempt < len(_LLM_RETRY_BACKOFF):
            await asyncio.sleep(_LLM_RETRY_BACKOFF[attempt])

    def _record_call(self, model: str | None, t0: float, ok: bool) -> None:
        """LLM 调用结果入指标：调用量{model,ok} + 延迟 summary（排障：失败率/慢调用可查）。"""
        m = model or settings.deepseek_model_chat
        metrics.inc("llm_calls", {"model": m, "ok": "true" if ok else "false"})
        metrics.observe("llm_latency_seconds", time.monotonic() - t0, {"model": m})

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
        thinking: bool | None = None,
    ) -> dict:
        """兼容旧 DeepSeekClient.chat 签名。Key 池化 + 排队 + 背压。

        temperature 可选：不传（None）不加该字段，走 DeepSeek 默认采样；
        意图分类等确定性场景传低温度（如 0.1）。
        tools/tool_choice 可选（工具决策循环用）：不传（None）则不加
        tools 字段，OpenAI 兼容格式原样透传给 DeepSeek，现有调用方无感。
        thinking 覆盖全局思考开关：None 用 settings.deepseek_thinking_enabled，
        False 关闭（意图分类/严重性评估等无 reasoning 消费方的调用省思考 token）。
        """
        if await self._breaker_open():
            # 熔断冷却期：入口直接快速失败，零网络尝试（防 LLM 慢挂时每请求 3 连打放大延迟）
            metrics.inc("llm_circuit_rejected")
            logger.warning("event=llm_circuit_rejected")
            raise LLMUnavailableError("LLM 服务暂时不可用，请稍后重试")
        try:
            # 排队（背压）：并发超过容量时等待，超时抛容量不足
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=settings.deepseek_queue_timeout,
            )
        except asyncio.TimeoutError:
            metrics.inc("llm_queue_timeout")
            logger.warning("event=queue_wait_timeout")
            raise CapacityExceededError("系统繁忙，请稍后再试") from None
        t0 = time.monotonic()
        try:
            result = await self._call(messages, model, timeout, temperature, tools, tool_choice, thinking)
            await self._breaker_reset()  # 一次逻辑调用成功（含换 Key 重试后成功）→ 重置计数
            self._record_call(model, t0, ok=True)
            return result
        except LLMUnavailableError:
            self._record_call(model, t0, ok=False)
            await self._breaker_fail()  # 网络/超时/重试耗尽 → 累计；AllKeysDown/Capacity 不累计
            raise
        except (AllKeysDownError, CapacityExceededError):
            # 无 healthy Key / 排队超时：调用未成功，计失败指标（非网络故障，不累计熔断）
            self._record_call(model, t0, ok=False)
            raise
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
        thinking: bool | None = None,
    ) -> dict:
        payload = {
            "model": model or settings.deepseek_model_chat,
            "messages": messages,
            "stream": False,
        }
        # 思考过程开关：与 _stream 一致，非流式响应在 message.reasoning_content 携带思考链
        #（决策循环直接作答路径依赖此字段透出 reasoning 事件）。
        # thinking 参数可覆盖全局开关：确定性分类/评估类调用传 False 省思考 token（无消费方）。
        if (settings.deepseek_thinking_enabled if thinking is None else thinking):
            payload["thinking"] = {"type": "enabled"}
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
                    await self._backoff_sleep(attempt)
                    continue  # 冷却后换 Key
                if resp.status_code >= 500:
                    logger.warning("event=llm_5xx", extra={"key_index": key.index, "status": resp.status_code})
                    await self._backoff_sleep(attempt)
                    continue  # 服务端错误换 Key
                resp.raise_for_status()  # 其他 4xx（参数错误等）不重试
            except httpx.TimeoutException:
                logger.error("event=llm_timeout", extra={"attempt": attempt})
                break  # 超时不重试
            except httpx.HTTPError as exc:
                logger.error("event=llm_http_error", extra={"attempt": attempt, "error": str(exc)})
                await asyncio.sleep(_LLM_RETRY_BACKOFF[min(attempt, len(_LLM_RETRY_BACKOFF) - 1)])
                continue

        raise LLMUnavailableError("LLM 调用失败，请稍后重试")

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[tuple[str, dict | None, str | None]]:
        """流式 chat（契约：token 级流式透出）。保留 Key 池化/排队/背压语义。

        yield (delta_content, usage_or_None, reasoning_or_None)：delta 非空为内容增量；
        usage 仅在最后一个 chunk（include_usage）出现，通常为 None；
        reasoning 为思考链增量（开启 thinking 时出现，通常为 None，先于 content 流出）。

        重试边界（决策：首个 content/reasoning delta 前可换 Key，之后绝不重试）：
        - HTTP 层 429/5xx → 换 Key 重试（未进内容流，安全）
        - 流内首个 content/reasoning delta 之前的异常 → 换 Key 重试
        - 首个 content/reasoning delta 之后的任何异常 → 直接上抛，不回滚已流出 token
        """
        if await self._breaker_open():
            metrics.inc("llm_circuit_rejected")
            logger.warning("event=llm_circuit_rejected")
            raise LLMUnavailableError("LLM 服务暂时不可用，请稍后重试")
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=settings.deepseek_queue_timeout,
            )
        except asyncio.TimeoutError:
            metrics.inc("llm_queue_timeout")
            logger.warning("event=queue_wait_timeout")
            raise CapacityExceededError("系统繁忙，请稍后再试") from None
        t0 = time.monotonic()
        try:
            async for item in self._stream(messages, model, timeout, temperature):
                yield item
            await self._breaker_reset()  # 正常流结束 → 成功
            self._record_call(model, t0, ok=True)
        except StreamInterruptedError:
            # 已产出首个 delta 后的流中断：连接级抖动/用户断连，非网关整体故障，不累计熔断
            # （挑战1：单次长流中途断不应误熔断全网关）。仍冒泡给上层规则引擎兜底。
            self._record_call(model, t0, ok=False)
            raise
        except LLMUnavailableError:
            self._record_call(model, t0, ok=False)
            await self._breaker_fail()
            raise
        finally:
            self._semaphore.release()

    async def _stream(
        self,
        messages: list[dict],
        model: str | None,
        timeout: float | None,
        temperature: float | None = None,
    ) -> AsyncIterator[tuple[str, dict | None, str | None]]:
        payload = {
            "model": model or settings.deepseek_model_chat,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},  # 最后一个 chunk 携带 usage
        }
        # 思考过程开关：deepseek-chat 默认不返回 reasoning_content，开启 thinking 才输出
        #（reasoning 事件依赖）。开启会额外计费思考 token。条件构建避免传无效参数被 API 拒绝。
        if settings.deepseek_thinking_enabled:
            payload["thinking"] = {"type": "enabled"}
        if temperature is not None:
            payload["temperature"] = temperature
        request_timeout = timeout or settings.deepseek_timeout_chat

        for attempt in range(3):  # 换 Key 最多重试 2 次
            key = await self._pool.select_key()
            if key is None:
                if await self._pool.all_cooling():
                    raise AllKeysDownError("所有 DeepSeek Key 均不可用")
                raise CapacityExceededError("系统繁忙，请稍后再试")

            started = False  # 是否已流出首个 content/reasoning delta（含 thinking 的思考链）
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
                            await self._backoff_sleep(attempt)
                            continue
                        if resp.status_code >= 500:
                            logger.warning(
                                "event=llm_5xx", extra={"key_index": key.index, "status": resp.status_code}
                            )
                            await self._backoff_sleep(attempt)
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
                            }, None
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        reasoning = delta.get("reasoning_content") or ""
                        if reasoning:
                            # 思考链增量：先于 content 流出；已流出则不可换 Key 重试（会重复思考内容）
                            started = True
                            yield "", None, reasoning
                        content = delta.get("content") or ""
                        if content:
                            started = True
                            yield content, None, None
                    return  # 正常流结束
            except httpx.TimeoutException:
                logger.error(
                    "event=llm_stream_timeout", extra={"attempt": attempt, "started": started}
                )
                if started:
                    raise StreamInterruptedError("LLM 流式中断") from None
                continue  # 未流出内容，可换 Key 重试
            except httpx.HTTPError as exc:
                logger.error(
                    "event=llm_stream_http_error",
                    extra={"attempt": attempt, "started": started, "error": str(exc)},
                )
                if started:
                    raise StreamInterruptedError("LLM 流式中断") from None
                await asyncio.sleep(_LLM_RETRY_BACKOFF[min(attempt, len(_LLM_RETRY_BACKOFF) - 1)])
                continue

        raise LLMUnavailableError("LLM 调用失败，请稍后重试")
