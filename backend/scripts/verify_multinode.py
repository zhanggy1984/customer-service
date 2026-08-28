"""多副本状态外置验证（直连各副本，绕过统一网关）。

前置：`docker compose up -d --scale backend=3`，取 3 副本容器 IP：
  docker inspect -f '{{.Name}} {{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' $(docker compose ps -q backend)

用法（探针容器，复用 backend 镜像全部依赖，挂载源码保证 import app.*）：
  docker run --rm --network shared-infra_shared-infra \
    -v D:/study/aiprojcet/customer-service/backend:/app -w /app \
    customer-service-backend \
    python scripts/verify_multinode.py --ips 172.x.x.1,172.x.x.2,172.x.x.3

验证 4 层（确定性断言，不依赖真实 LLM——订单查询走规则短路）：
  1. 部署层：3 副本 /healthz 均 200（多副本可达）
  2. 锁互斥（跨进程）：multiprocessing 两个独立进程各自 RedisSessionLock(sid)——
     A 持锁期间 B 必须等待（非立即成功），A 释放后 B 可接力。两个进程 = 两个节点本质
  3. 端到端冒烟：并发对 3 副本发同一 session 订单查询，断言均完成无 5xx
  4. 熔断广播：SET cs:cb:db:open → _breaker_open()=True（他节点信号被读取）
     → HTTP 触发 DB 调用快速失败 → 清理信号后恢复 False
"""
import argparse
import asyncio
import json
import multiprocessing as mp
import sys
import time

import httpx
import redis.asyncio as aioredis

sys.path.insert(0, ".")  # 探针容器 WORKDIR=/app，挂载 backend 源码
from app.config import settings
from app.session.locks import RedisSessionLock, SessionLockTimeoutError
from app.services.retry import _breaker_open

ORDER_PROMPT = "查询订单 ORD-20240801-001 的状态"  # 规则短路：订单号确定性直查，不调 LLM
BREAKER_KEY = "cs:cb:db:open"
LOCK_SID = "verify-lock-session"

PASS = 0
FAIL = 0


def _report(name: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    mark = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{mark}] {name}{' — ' + detail if detail else ''}")
    return ok


# ---------- 1. 部署层（同步 httpx 直连各副本） ----------

def verify_deploy(ips: list[str]) -> bool:
    print("[1/4] 部署层：3 副本可达")
    all_ok = True
    for ip in ips:
        try:
            r = httpx.get(f"http://{ip}:8000/healthz", timeout=5)
            ok = r.status_code == 200
            _report(f"{ip}:8000 /healthz", ok, r.text[:50] if not ok else "")
            all_ok &= ok
        except Exception as exc:
            _report(f"{ip}:8000 /healthz", False, str(exc)[:80])
            all_ok = False
    return all_ok


# ---------- 2. 锁互斥（跨进程，multiprocessing 放 async 外） ----------

def _lock_holder(sid: str, held: mp.Event, release: mp.Event) -> None:
    """持锁方：acquire 成功通知主进程，等待 release 后释放。

    mp.Event.wait 是同步阻塞（不能 await），放 asyncio 之外。持锁期间无其他
    协程需要调度，进程级阻塞安全。
    """
    async def _acquire() -> RedisSessionLock:
        lock = RedisSessionLock(sid)
        await lock._acquire()
        return lock

    lock = asyncio.run(_acquire())
    held.set()
    release.wait(10)
    asyncio.run(lock._release())


def _lock_waiter(sid: str, start: mp.Event, result_q: mp.Queue) -> None:
    """等锁方：A 已持锁后尝试 acquire，应阻塞等待而非立即成功。"""
    start.wait()  # 同步等待主进程发令（A 已持锁）

    async def _main():
        t0 = time.monotonic()
        lock = RedisSessionLock(sid)
        try:
            await lock._acquire()
            result_q.put(("acquired", round(time.monotonic() - t0, 3)))
            await lock._release()
        except SessionLockTimeoutError:
            result_q.put(("timeout", round(time.monotonic() - t0, 3)))

    asyncio.run(_main())


def verify_lock_multiprocess() -> bool:
    print("[2/4] 锁互斥：跨进程 RedisSessionLock 原子互斥")
    ctx = mp.get_context("fork")
    held, release, start = ctx.Event(), ctx.Event(), ctx.Event()
    result_q = ctx.Queue()
    p_holder = ctx.Process(target=_lock_holder, args=(LOCK_SID, held, release))
    p_waiter = ctx.Process(target=_lock_waiter, args=(LOCK_SID, start, result_q))
    p_holder.start()
    if not held.wait(10):
        p_holder.terminate()
        return _report("A 未能在 10s 内持锁（Redis 不可达？）", False)
    p_waiter.start()
    start.set()
    time.sleep(1.5)          # 让 B 进入等待循环（A 仍持锁）
    release.set()            # 放行 A 释放 → B 应接力拿到
    kind, elapsed = result_q.get(timeout=15)
    p_holder.join(8)
    p_waiter.join(8)
    # B 要么等到 A 释放（acquired 且耗时>=0.5s），要么锁等待超时——都证明非"立即成功"
    ok = (kind == "timeout") or (kind == "acquired" and elapsed >= 0.5)
    return _report(f"B acquire 结果 {kind}（耗时 {elapsed}s）", ok,
                   "B 未立即拿到锁 = 同 session 被 A 独占" if ok else "B 立即拿到锁 = 锁未互斥")


# ---------- HTTP 工具（直连副本） ----------

def _parse_sse(body: bytes) -> list[dict]:
    events = []
    for line in body.decode("utf-8").split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:]))
            except json.JSONDecodeError:
                pass
    return events


async def _login(base: str) -> str:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{base}/auth/login", json={"username": "user_1", "password": "123456"})
        assert r.status_code == 200, f"登录失败: {r.status_code} {r.text[:200]}"
        return r.json()["access_token"]


async def _create_session(base: str, token: str) -> str:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{base}/sessions", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 201, f"建会话失败: {r.status_code} {r.text[:200]}"
        return r.json()["session_id"]


async def _chat(base: str, token: str, sid: str, content: str) -> tuple[int, list[dict]]:
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(
            f"{base}/sessions/{sid}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"content": content},
        )
        return r.status_code, _parse_sse(r.content)


# ---------- 3. 端到端冒烟（3 副本并发同 session） ----------

async def verify_end_to_end(ips: list[str], token: str, sid: str) -> bool:
    print("[3/4] 端到端：3 副本并发同一 session 订单查询")
    results = await asyncio.gather(*[_chat(f"http://{ip}:8000/api/v1", token, sid, ORDER_PROMPT)
                                    for ip in ips])
    all_ok = True
    for ip, (st, events) in zip(ips, results):
        ok = st < 500 and len(events) > 0
        evt_types = [e.get("type") for e in events[-2:]]
        _report(f"{ip} 响应 {st}（事件 {len(events)} 个，末 {evt_types}）", ok)
        all_ok &= ok
    return all_ok


# ---------- 4. 熔断广播（信号共享 + HTTP 快速失败） ----------

async def verify_breaker_broadcast(ips: list[str], token: str, sid: str) -> bool:
    print("[4/4] 熔断广播：cs:cb:db:open 信号跨节点读取")
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    await redis.delete(BREAKER_KEY)
    await asyncio.sleep(0.3)
    base_ok = (await _breaker_open()) is False

    await redis.set(BREAKER_KEY, "1", ex=60)
    await asyncio.sleep(0.3)
    shared_ok = (await _breaker_open()) is True

    st, events = await _chat(f"http://{ips[0]}:8000/api/v1", token, sid, ORDER_PROMPT)
    fast_fail_ok = st < 500  # 熔断期 DB 调用快速失败，无重试风暴/不挂起

    await redis.delete(BREAKER_KEY)
    await asyncio.sleep(0.3)
    restored_ok = (await _breaker_open()) is False
    await redis.aclose()

    return (_report("基线：无信号不熔断", base_ok)
            and _report("写入信号后 _breaker_open()=True", shared_ok)
            and _report(f"熔断期 HTTP {st}（快速失败）", fast_fail_ok)
            and _report("清理信号后恢复 False", restored_ok))


# ---------- main ----------

async def _async_stage(ips: list[str]):
    token = await _login(f"http://{ips[0]}:8000/api/v1")
    sid = await _create_session(f"http://{ips[0]}:8000/api/v1", token)
    ok3 = await verify_end_to_end(ips, token, sid)
    ok4 = await verify_breaker_broadcast(ips, token, sid)
    return ok3, ok4


def main() -> int:
    ap = argparse.ArgumentParser(description="多副本状态外置验证（直连）")
    ap.add_argument("--ips", required=True, help="3 副本容器 IP，逗号分隔")
    args = ap.parse_args()
    ips = [i.strip() for i in args.ips.split(",") if i.strip()]
    if len(ips) < 2:
        print(f"[ERROR] 至少需要 2 个副本，当前 {len(ips)} 个")
        return 2

    ok1 = verify_deploy(ips)
    ok2 = verify_lock_multiprocess()
    ok3, ok4 = asyncio.run(_async_stage(ips))

    print(f"\n结果: {PASS} PASS / {FAIL} FAIL（部署={ok1} 锁互斥={ok2} 端到端={ok3} 熔断广播={ok4}）")
    return 0 if (ok1 and ok2 and ok3 and ok4 and FAIL == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
