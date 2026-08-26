"""意图分类器。

classify_intent() 调用 deepseek-chat 做 6 分类，返回 IntentResult。
JSON 容错：_extract_json 去除 markdown 包裹 → json.loads → 2 次重试 → 仍失败降级 CHITCHAT。
"""
import json
import re
from dataclasses import dataclass, field

from app.agent.intent_rules import match_intent_rules
from app.agent.prompts.guard import guard_user_content
from app.agent.prompts.intent import build_intent_system
from app.config import settings
from app.infrastructure.deepseek import deepseek_client
from app.utils.logger import logger

# 规则命中视为确定性分类（> SWITCH_THRESHOLD=0.8），业务流内规则已禁用，不会触发意外切换
RULE_HIT_CONFIDENCE = 0.97

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
    usage: dict | None = None  # 本轮分类调用的 token 消耗（计入聚合 usage）


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
    injection_detected: bool = False,
    use_rules: bool = True,
) -> IntentResult:
    # 规则前置短路：高置信模板化表达不调 LLM。
    # 禁用场景：injection_detected（规则会跳过注入防御声明）、业务流内（orchestrator 传 False，
    # 流内短词"确认/好的/补充"本身模糊，规则必误判，须保留 LLM + state_hint）。
    if use_rules and not injection_detected:
        hit = match_intent_rules(user_input)
        if hit:
            logger.info(
                "event=intent_rule_hit",
                extra={"intent": hit.intent, "input_len": len(user_input)},
            )
            return IntentResult(
                intent=hit.intent,
                confidence=RULE_HIT_CONFIDENCE,
                slots=hit.slots,
                missing_slots=hit.missing_slots,
                summary=hit.summary,
                usage=None,  # 未调 LLM，无 token 消耗
            )
    # 用户输入放独立 user 消息（不拼进 system，消除注入面）；命中注入时前置防御声明
    messages = [
        {"role": "system", "content": build_intent_system(current_state_context)},
        {"role": "user", "content": guard_user_content(user_input, injection_detected)},
    ]

    for attempt in range(max_retries):
        try:
            data = await deepseek_client.chat(
                messages,
                model=settings.deepseek_model_chat,
                temperature=0.1,  # 意图分类需确定性，低温度抑制同 query 分类抖动
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
                usage=data.get("usage"),
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
            # 追加严格 JSON 提示重试（改 system 消息本体，user 消息不动）
            messages[0]["content"] += "\n[注意] 必须输出严格的 JSON，不要用 ```json 包裹。"

    return _chitchat_fallback()
