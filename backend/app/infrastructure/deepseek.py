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
