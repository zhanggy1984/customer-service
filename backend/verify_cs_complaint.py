"""verify_cs_complaint.py：投诉状态机 reasoning/tool_call 透出验证。

绕过意图分类（避免 LLM 分类漂移干扰契约验证），直接以 COMPLAINT 状态
推进状态机，断言 severity_assess 节点透出 reasoning、execute 节点透出
tool_call(create_complaint)。在容器内运行。
"""
import asyncio

import httpx

from app.agent import usage
from app.agent.state_machine.complaint_flow import ComplaintFlow
from app.infrastructure.mysql import mysql_pool

BASE = "http://localhost:8000/api/auth"  # T15：登录路由统一 /api/auth/login


async def _get_user_id() -> int:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(BASE + "/login", json={"username": "cs_verify", "password": "verify123456"})
        r.raise_for_status()
        return r.json()["user"]["id"]


async def main() -> None:
    uid = await _get_user_id()
    await mysql_pool.init()  # 模拟应用 lifespan 初始化（脚本直连无 FastAPI 启动）
    usage.begin()
    flow = ComplaintFlow()
    # 直接置于 collect_description：输入描述即推进 severity_assess → execute → notify
    st = {
        "user_id": uid,
        "session_id": "unit-complaint",
        "order_id": None,
        "complaint_type": "商品质量",
        "stage": "collect_description",
        "awaiting": "description",
        "description": "",
    }
    st = await flow.step(st, "我买的手机屏幕碎了，明显是质量问题")
    print("stage:", st.get("stage"), "| final:", st.get("final"), "| severity:", st.get("severity"))
    print("reasoning:", st.get("reasoning"))
    print("tool_calls:", st.get("tool_calls"))
    print("message:", st.get("message"))
    print("usage:", usage.current())

    assert st.get("reasoning"), "❌ 缺 reasoning（severity 评估未透出）"
    assert st.get("tool_calls"), "❌ 缺 tool_calls（create_complaint 未透出）"
    assert st.get("tool_calls")[0]["name"] == "create_complaint", "❌ tool 名不符"
    assert st.get("final"), "❌ 未到终态"
    print("✅ 投诉状态机 reasoning + tool_call(create_complaint) 透出正确")


if __name__ == "__main__":
    asyncio.run(main())
