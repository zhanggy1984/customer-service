"""会话领域模型。序列化为 JSON 存 Redis（Phase 4 由 StorageRouter 加 MySQL 兜底）。"""
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Message(BaseModel):
    role: str  # user | assistant
    content: str
    ts: datetime = Field(default_factory=_utcnow)


class Session(BaseModel):
    session_id: str
    user_id: int
    messages: list[Message] = Field(default_factory=list)
    # Agent 状态机上下文：{intent, agent_state, ...}，进行中的流程靠它续推
    agent_state: dict | None = None
    intent: str | None = None
    # 意图切换快照：{intent: state}，中途切走时保存，可恢复
    snapshots: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def to_llm_messages(self, limit: int | None = None) -> list[dict]:
        """转成 LLM 需要的消息列表（截断最近 limit 条）。"""
        msgs = self.messages[-limit:] if limit else self.messages
        return [{"role": m.role, "content": m.content} for m in msgs]
