"""verify_cs_e2e.py：customer-service 契约改造 e2e 验证。

覆盖契约事件（§5.1）：meta/status/reasoning/tool_call/token.delta/usage/done/error。
- 场景 1 闲聊（CHITCHAT）    ：流式 token.delta 多帧 + usage
- 场景 2 订单查询（ORDER_STATUS）：tool_call(query_order) + 静态 token（未找到路径）
- 场景 3 政策查询（POLICY_INQUIRY）：tool_call(search_policy) + 流式 token
- 场景 4 投诉（COMPLAINT）   ：reasoning（severity 评估）+ tool_call(create_complaint) + 静态 token

在容器内运行：docker compose exec -T backend python verify_cs_e2e.py
容忍 LLM 意图分类漂移：打印实际事件序列，仅对「已发生的事件」做契约结构断言。
"""
import asyncio
import json

import httpx

# T15 契约：auth 挂 /api/auth/*，sessions 挂 /api/v1/*，故 BASE=/api 再按段拼接
BASE = "http://localhost:8000/api"
USER = "cs_verify"
PASS = "verify123456"


async def _post(path: str, body: dict, token: str | None = None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=30) as c:
        return await c.post(f"{BASE}{path}", json=body, headers=headers)


async def _register_login() -> str:
    # 幂等：已存在则 409 忽略
    r = await _post("/auth/register", {"username": USER, "password": PASS})
    if r.status_code not in (201, 409):
        raise RuntimeError(f"register 失败: {r.status_code} {r.text}")
    r = await _post("/auth/login", {"username": USER, "password": PASS})
    r.raise_for_status()
    return r.json()["access_token"]


async def _new_session(token: str) -> str:
    r = await _post("/v1/sessions", {}, token)
    r.raise_for_status()
    return r.json()["session_id"]


async def _send(token: str, sid: str, content: str) -> list[dict]:
    """POST 消息，解析 SSE 帧 → [{type, data}]。"""
    events: list[dict] = []
    async with httpx.AsyncClient(timeout=90) as c:
        async with c.stream(
            "POST",
            f"{BASE}/v1/sessions/{sid}/messages",
            json={"content": content},
            headers={"Authorization": f"Bearer {token}"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    events.append({"type": line[6:].strip(), "data": None})
                elif line.startswith("data:") and events:
                    events[-1]["data"] = json.loads(line[5:].strip())
    return events


def _summary(events: list[dict]) -> str:
    """事件序列摘要：type + 关键字段。"""
    parts = []
    for ev in events:
        d = ev["data"] or {}
        t = ev["type"]
        if t == "token":
            parts.append(f"token[{len(d.get('delta',''))}字]")
        elif t == "tool_call":
            parts.append(f"tool_call({d.get('name')}={d.get('status')})")
        elif t == "usage":
            parts.append(f"usage(total={d.get('total_tokens')})")
        elif t == "status":
            parts.append(f"status({d.get('stage')})")
        else:
            parts.append(t)
    return " → ".join(parts)


def _check(events: list[dict], name: str, required: dict) -> None:
    """断言某类事件存在且结构符合契约。required: {type: [字段] | None}。"""
    by_type: dict[str, list[dict]] = {}
    for ev in events:
        by_type.setdefault(ev["type"], []).append(ev["data"] or {})
    missing = [t for t in required if t not in by_type]
    print(f"  [{name}] 事件: {_summary(events)}")
    if missing:
        print(f"  ❌ 缺少事件: {missing}")
        return
    for t, fields in required.items():
        if fields is None:
            print(f"  ✅ {t} 存在")
            continue
        bad = [f for f in fields if f not in by_type[t][0]]
        # usage/token 允许多帧，检查首帧即可；必选字段缺则报
        if bad:
            print(f"  ❌ {t} 缺字段 {bad}: {by_type[t][0]}")
        else:
            print(f"  ✅ {t} 字段齐全 {fields}")
    # ts 内置校验（契约：data 内置 ts，agent 侧 unix ms）
    no_ts = [t for t in events if "ts" not in (t["data"] or {})]
    if no_ts:
        names = ", ".join(n["type"] for n in no_ts)
        print(f"  ❌ 缺 ts 的事件: {names}")
    else:
        print("  ✅ 全部事件含 ts")
    # done 校验
    if "done" in by_type and "usage" in by_type:
        print("  ✅ done/usage 必选事件齐")


async def _scene(token: str, sid: str, name: str, content: str, expect: str) -> list[dict]:
    print(f"\n===== 场景 {name}：{content}（期望 {expect}） =====")
    events = await _send(token, sid, content)
    # done 在最后；token 拼接应与 done.content 一致
    answers = "".join(e["data"]["delta"] for e in events if e["type"] == "token")
    done = next((e["data"] for e in events if e["type"] == "done"), None)
    if done is not None and answers:
        done_content = done.get("content", "")
        ok = answers == done_content
        if ok:
            print("  ✅ token 拼接 == done.content")
        else:
            print(f"  ❌ token 拼接≠done.content: 拼={answers[:30]!r} done={done_content[:30]!r}")
    elif done is not None and not answers:
        print("  ⚠️ 无 token 事件（规则/静态话术路径？done.content 直接提供）")
    return events


async def main() -> None:
    token = await _register_login()
    sid = await _new_session(token)
    print(f"已注册/登录用户 {USER}，新建会话 {sid}")

    # 场景 1：闲聊（期望流式 token）
    ev1 = await _scene(token, sid, "1.闲聊", "你好呀，能介绍一下你能做什么吗", "CHITCHAT")
    _check(ev1, "闲聊契约", {
        "meta": ["agent", "model", "interface", "contract_version"],
        "token": ["content", "delta"],
        "usage": ["prompt_tokens", "completion_tokens", "total_tokens"],
        "done": None,
    })

    # 场景 2：订单查询（期望 tool_call）
    ev2 = await _scene(token, sid, "2.订单", "查询订单 999999999", "ORDER_STATUS")
    _check(ev2, "订单契约", {
        "tool_call": ["id", "name", "args", "result", "status"],
        "usage": None,
        "done": None,
    })

    # 场景 3：政策查询（期望 tool_call + 流式 token）
    ev3 = await _scene(token, sid, "3.政策", "你们的退货政策是什么", "POLICY_INQUIRY")
    _check(ev3, "政策契约", {
        "tool_call": ["name", "status"],
        "token": ["content", "delta"],
        "usage": None,
        "done": None,
    })

    # 场景 4：投诉（两轮，期望 reasoning + tool_call(create_complaint)）
    await _scene(token, sid, "4a.投诉-发起", "我要投诉，你们商品质量太差了", "COMPLAINT")
    ev4 = await _scene(token, sid, "4b.投诉-描述", "我买的手机屏幕碎了，明显质量问题", "COMPLAINT")
    _check(ev4, "投诉契约", {
        "reasoning": ["content", "delta"],
        "tool_call": ["name", "status"],
        "usage": None,
        "done": None,
    })

    print("\n===== 验证完成 =====")


if __name__ == "__main__":
    asyncio.run(main())
