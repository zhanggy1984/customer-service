"""DeepSeek 客户端（Phase 4 起由 Gateway 实现 Key 池化 + 排队 + 背压）。

对外暴露 deepseek_client.chat(messages, model, timeout) 兼容接口，
调用方（intent/orchestrator/complaint）无需改动。
异常: CapacityExceededError / AllKeysDownError / LLMUnavailableError
"""
from app.infrastructure.deepseek_gateway import (
    AllKeysDownError,
    CapacityExceededError,
    DeepSeekGateway,
    LLMUnavailableError,
)

deepseek_client = DeepSeekGateway()

# 熔断降级判定集合（统一出口，agent 侧 orchestrator/agent_loop 复用，避免两处重复定义）。
# 语义：这三种异常代表"LLM 不可用/排队满/全 Key 冷"，上层 _rule_engine_fallback
# 捕获后降级规则引擎；非 LLM 异常（解析/编程错误）不在此列，走各自的兜底路径。
LLM_FALLBACK_ERRORS = (LLMUnavailableError, CapacityExceededError, AllKeysDownError)
