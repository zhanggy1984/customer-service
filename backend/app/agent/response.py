"""SSE 响应协议。事件格式: {"type": "status"|"token"|"done"|"error", ...}。"""
import json


def sse_format(evt: dict) -> str:
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"


def status_event(stage: str, message: str) -> dict:
    return {"type": "status", "stage": stage, "message": message}


def token_event(content: str) -> dict:
    return {"type": "token", "content": content}


def done_event(intent: str) -> dict:
    return {"type": "done", "intent": intent}


def error_event(message: str) -> dict:
    return {"type": "error", "message": message}
