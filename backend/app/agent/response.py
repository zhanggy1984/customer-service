"""SSE 响应协议（评测契约 §5.1）。

帧格式: event: <type>\ndata: <json>\n\n，data 内置 ts（unix ms，agent 侧生成）。
事件 type: meta/stage/reasoning/tool_call/answer/usage/done/error，另保留旧 status/action 兼容前端。
answer/reasoning 为 delta 增量，评测端拼接；usage/done 必选。

向后兼容：旧前端 useSSE.ts 只读 data 行并按 json.type 分发，新增 event: 帧与
新增事件类型对其透明（未知 type 静默忽略，done 仍携带 content）。
"""
import json
import time


def sse_format(evt: dict) -> str:
    """序列化 SSE 帧。data 内置 ts（agent 侧 unix ms）。"""
    evt = {**evt, "ts": int(time.time() * 1000)}
    return f"event: {evt['type']}\ndata: {json.dumps(evt, ensure_ascii=False)}\n\n"


def status_event(stage: str, message: str) -> dict:
    return {"type": "status", "stage": stage, "message": message}


def token_event(content: str) -> dict:
    return {"type": "token", "content": content}


def meta_event(meta: dict) -> dict:
    """会话元数据（可选）。字段: agent/model/interface/contract_version/git_sha/knowledge_version。"""
    return {"type": "meta", **meta}


def answer_event(delta: str) -> dict:
    """终答增量。评测端首个 answer.delta 即 TTFT 起点（§5.3）。"""
    return {"type": "answer", "delta": delta}


def reasoning_event(delta: str) -> dict:
    """思考链增量（可选）。"""
    return {"type": "reasoning", "delta": delta}


def tool_call_event(call: dict) -> dict:
    """观测层外显工具调用（决策 #8，不强制 LLM function calling）。"""
    return {"type": "tool_call", **call}


def usage_event(usage: dict) -> dict:
    """Token 消耗（必选）。一轮对话聚合单条。"""
    return {"type": "usage", **usage}


def done_event(intent: str) -> dict:
    return {"type": "done", "intent": intent}


def error_event(message: str) -> dict:
    return {"type": "error", "message": message}
