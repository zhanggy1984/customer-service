"""意图分类器。

classify_intent() 调用 deepseek-chat 做 6 分类，返回 IntentResult。
JSON 容错：_extract_json 去除 markdown 包裹 → json.loads → 2 次重试 → 仍失败降级 CHITCHAT。
"""
import json
import re
from dataclasses import dataclass, field

from app.agent.prompts.intent import build_intent_prompt
from app.config import settings
from app.infrastructure.deepseek import deepseek_client
from app.utils.logger import logger

VALID_INTENTS = {
    "POLICY_INQUIRY",
    "ORDER_STATUS",
    "RETURN_REQUEST",
    "REFUND_REQUEST",
    "COMPLAINT",
    "CHITCHAT",
}


@dataclass
class IntentResult:
    intent: str = "CHITCHAT"
    confidence: float = 0.0
    slots: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    summary: str = ""


def _extract_json(text: str) -> str:
    """从 LLM 输出中提取 JSON。兼容 markdown 代码块包裹 / 前后尾随文本。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def _chitchat_fallback() -> IntentResult:
    return IntentResult(intent="CHITCHAT", confidence=0.0, summary="意图分类失败，兜底为闲聊")


async def classify_intent(
    user_input: str,
    current_state_context: str | None = None,
    max_retries: int = 2,
) -> IntentResult:
    prompt = build_intent_prompt(user_input, current_state_context)

    for attempt in range(max_retries):
        try:
            data = await deepseek_client.chat(
                [{"role": "system", "content": prompt}],
                model=settings.deepseek_model_chat,
            )
            raw = data["choices"][0]["message"]["content"]
            json_str = _extract_json(raw)
            parsed = json.loads(json_str)

            result = IntentResult(
                intent=parsed.get("intent", "CHITCHAT"),
                confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.0)))),
                slots=parsed.get("slots", {}) or {},
                missing_slots=parsed.get("missing_slots", []) or [],
                summary=parsed.get("summary", "") or "",
            )
            if result.intent not in VALID_INTENTS:
                result.intent = "CHITCHAT"

            logger.info(
                "event=intent_classified",
                extra={
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "attempt": attempt,
                    "input_len": len(user_input),
                },
            )
            return result
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, IndexError):
            if attempt == max_retries - 1:
                logger.warning(
                    "event=intent_parse_fallback",
                    extra={"input_len": len(user_input), "attempt": attempt},
                )
                return _chitchat_fallback()
            # 追加严格 JSON 提示重试
            prompt += "\n[注意] 必须输出严格的 JSON，不要用 ```json 包裹。"

    return _chitchat_fallback()
