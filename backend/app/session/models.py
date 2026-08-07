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

    def trim(self, max_messages: int) -> None:
        """消息体截断：超过 max_messages 条时，保留首条 user 消息 + 最近 N-1 条。

        首条 user 消息是会话列表标题的锚点（list_sessions 取它做标题），删掉会导致标题退化；
        其余更早消息是截断的物理丢弃（前端历史仅展示保留部分）。状态机续推靠 agent_state，
        不依赖 messages 全文，故截断不影响业务。
        """
        if len(self.messages) <= max_messages:
            return
        head = [self.messages[0]] if self.messages and self.messages[0].role == "user" else []
        keep = max_messages - len(head)
        # keep 可能为 0（max=1 且保留首条后无剩余空间），此时 tail 必须为空；
        # 直接 [-0:] 等价 [:] 会返回整个列表，造成首条重复且不截断
        tail = self.messages[-keep:] if keep > 0 else []
        self.messages = head + tail

    def to_llm_messages(self, limit: int | None = None) -> list[dict]:
        """转成 LLM 需要的消息列表（截断最近 limit 条）。"""
        msgs = self.messages[-limit:] if limit else self.messages
        return [{"role": m.role, "content": m.content} for m in msgs]
