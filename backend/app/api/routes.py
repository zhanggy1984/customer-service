"""业务接口：会话创建 + 消息发送 + Admin 管理。

- POST /api/v1/sessions               创建会话，返回 session_id
- POST /api/v1/sessions/{sid}/messages 发送消息，返回 LLM 回复（SSE 流式）
- Admin: 知识库管理（MySQL 为源 + ChromaDB 同步，见 rag/kb_store.py）+ 订单维护
"""
import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent import usage
from app.agent.orchestrator import run_agent
from app.agent.response import answer_event, sse_format, usage_event
from app.api.deps import get_current_user, require_admin
from app.config import settings
from app.infrastructure.mysql import mysql_pool
from app.rag import kb_store
from app.session.locks import session_locks
from app.session.manager import session_manager
from app.session.models import Message
from app.utils.logger import logger

router = APIRouter(tags=["chat"])  # 无 prefix，挂载时由 main.py 统一加 /api/v1


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


@router.post("/sessions", status_code=201)
async def create_session(user: dict = Depends(get_current_user)) -> dict:
    session = await session_manager.create_session(user_id=int(user["sub"]))
    logger.info("event=api_create_session", extra={"session_id": session.session_id, "user_id": user["sub"]})
    return {"session_id": session.session_id}


@router.post("/sessions/{sid}/messages")
async def send_message(
    sid: str,
    req: SendMessageRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """发送消息，返回 SSE 流（status → done/error）。

    同一 session 的并发请求通过 SessionLocks 串行化（Phase 4.5），
    锁内重新加载最新会话状态，防止并发读写导致状态机覆盖。
    """
    lock = await session_locks.get(sid)

    async def event_gen():
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(evt: dict) -> None:
            await queue.put(evt)

        # 首帧 meta（契约 §5.1，可选）：agent/model/interface/contract_version。
        # git_sha/knowledge_version 当前无版本管理机制，透出空串。
        await emit({
            "type": "meta",
            "agent": "customer-service",
            "model": settings.deepseek_model_chat,
            "interface": "sessions/{sid}/messages",
            "contract_version": "1.0",
            "git_sha": "",
            "knowledge_version": "",
        })

        async def run_and_finish() -> None:
            async with lock:  # 同一 session 串行
                session = await session_manager.get_session(sid)
                created_new = False
                if session is None:
                    session = await session_manager.create_session(int(user["sub"]))
                    created_new = True
                elif session.user_id != int(user["sub"]):
                    await emit({"type": "error", "message": "无权访问该会话"})
                    return
                session.messages.append(Message(role="user", content=req.content))
                try:
                    if created_new:
                        reply = "您好！请问有什么可以帮您？可以查询订单、退货、退款等。"
                        # 契约 §5.1：greeting 无 LLM 调用，answer 全量补发（评测端首个 answer.delta 即 TTFT
                        # 起点，且按 answer.delta 拼接最终回复）+ usage 补发（字段齐全）。
                        await emit(answer_event(reply))
                        await emit(usage_event(usage.current()))
                    else:
                        reply = await run_agent(session, req.content, int(user["sub"]), emit)
                    session.messages.append(Message(role="assistant", content=reply))
                    await session_manager.update_session(session)
                except Exception as exc:
                    logger.error("event=agent_error", extra={"session_id": sid, "error": str(exc)})
                    await emit({"type": "error", "message": "系统出问题了，请稍后重试"})
                    return
                # done 事件携带完整回复文本 + 实际 session_id（会话重建时前端需更新）
                await emit({"type": "done", "intent": session.intent or "", "content": reply, "session_id": session.session_id})

        task = asyncio.create_task(run_and_finish())
        while True:
            if await request.is_disconnected():
                task.cancel()
                break
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                continue
            yield sse_format(evt)
            if evt["type"] in ("done", "error"):
                break
        if not task.done():
            await task

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 禁止 Nginx 缓冲，保证 SSE 逐帧推送
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)) -> dict:
    """当前用户的会话列表（供前端侧边栏展示 / 切换历史会话）。

    数据源：MySQL conversation_history（StorageRouter 每次 save 双写落库，含 user_id 索引）。
    标题取首条 user 消息，空会话（未发过消息）不展示，按写入顺序（id 自增）倒序取最近 50 个。
    """
    rows = await mysql_pool.fetchall(
        # JSON_LENGTH 判断消息非空数组，语义明确且避免 JSON 列与字符串 '' 比较的跨版本兼容风险
        "SELECT session_id, intent, messages, created_at FROM conversation_history "
        "WHERE user_id=%s AND JSON_LENGTH(messages) > 0 "
        "ORDER BY id DESC LIMIT 50",
        (int(user["sub"]),),
    )
    items: list[dict] = []
    for row in rows:
        title = "新会话"
        try:
            msgs = json.loads(row["messages"] or "[]")
            if not isinstance(msgs, list):  # 脏数据防御：非数组按空处理，避免逐字符遍历
                msgs = []
        except (json.JSONDecodeError, TypeError):
            msgs = []
        for m in msgs:
            # 数组内元素同样防御：非 dict 跳过、content 非字符串跳过，
            # 避免 m.get()/content[:30] 对非法元素抛异常导致整个列表 500
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if m.get("role") == "user" and isinstance(content, str) and content:
                title = content[:30]
                break
        # _save_mysql 每次 DELETE+INSERT，此时间实为"最后保存/活跃时间"，故命名为 updated_at
        updated_at = row["created_at"]
        if hasattr(updated_at, "isoformat"):
            updated_at = updated_at.isoformat()
        items.append(
            {
                "session_id": row["session_id"],
                "title": title,
                "updated_at": updated_at,
                "intent": row["intent"],
            }
        )
    logger.info("event=api_list_sessions", extra={"user_id": user["sub"], "count": len(items)})
    return {"items": items, "total": len(items)}


@router.get("/sessions/{sid}/messages")
async def get_session_messages(sid: str, user: dict = Depends(get_current_user)) -> dict:
    """拉取单个会话的历史消息（SSE 发消息前前端用于恢复历史渲染）。

    归属校验与 send_message 一致；会话从 Redis 过期（TTL）时经 storage_router 走 MySQL 兜底重建。
    """
    session = await session_manager.get_session(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session.user_id != int(user["sub"]):
        raise HTTPException(status_code=403, detail="无权访问该会话")
    messages = [m.model_dump(mode="json") for m in session.messages[-200:]]
    logger.info(
        "event=api_get_session_messages",
        extra={"session_id": sid, "user_id": user["sub"], "count": len(messages)},
    )
    return {"session_id": sid, "intent": session.intent, "messages": messages}


@router.delete("/sessions/{sid}")
async def delete_session(sid: str, user: dict = Depends(get_current_user)) -> dict:
    """删除单个会话（Redis + MySQL conversation_history 一并清除）。

    归属校验与历史读取一致：仅能删除自己的会话，admin 无跨用户特权。
    与 send_message 同锁串行化：防止删除后同会话并发 send 重新 save 复活（TOCTOU）。
    """
    lock = await session_locks.get(sid)
    async with lock:
        session = await session_manager.get_session(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        if session.user_id != int(user["sub"]):
            raise HTTPException(status_code=403, detail="无权删除该会话")
        await session_manager.close_session(sid)
    logger.info("event=api_delete_session", extra={"session_id": sid, "user_id": user["sub"]})
    return {"msg": "已删除"}


# =============================================================
# Admin 知识库管理（require_admin 校验）
# 设计：MySQL knowledge_docs 为源，ChromaDB 为派生索引，见 rag/kb_store.py
# 一致性：写 ChromaDB 失败 → sync_status=pending → 增量补偿 / 手动"同步"按钮全量对账
# =============================================================
class KnowledgeUploadRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1)


class KnowledgeUpdateRequest(BaseModel):
    content: str = Field(min_length=1)


KB_UPLOAD_EXTENSIONS = {".md", ".txt"}
KB_UPLOAD_MAX_BYTES = 1024 * 1024  # 1MB
_KB_UPLOAD_CHUNK_SIZE = 64 * 1024  # 流式读分块，防全量进内存


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """流式分块读取上传文件，累计超限即拒绝（内存峰值≈max_bytes+分块）。

    UploadFile.read() 一次性全量读入内存，先读后校验会让超大文件在 413 之前
    占满内存（multipart 解析阶段的 SpooledTemporaryFile 只挡解析不挡 handler）。
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_KB_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="文件过大，仅支持 1MB 以内")
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_knowledge_upload(filename: str, source_name: str, raw: bytes) -> tuple[str, str]:
    """校验并解析上传的知识库文件，返回 (source, content)。

    - 扩展名：按 filename 判断，仅 .md/.txt（纯文本；pdf/docx 需解析器，超出范围）→ 415
    - 编码：utf-8 解码失败 → 400
    - 内容：strip 后为空 → 400
    - source：source_name（表单 title，可空）否则文件名 stem，截断到 100（source VARCHAR(100)）
    """
    ext = Path(filename).suffix.lower()
    if ext not in KB_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"仅支持 {'/'.join(sorted(KB_UPLOAD_EXTENSIONS))} 文本文件")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="文件编码必须为 UTF-8")
    if not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")
    source = (source_name or Path(filename).stem).strip() or filename
    return source[:100], content


@router.post("/admin/knowledge", status_code=201)
async def upload_knowledge(req: KnowledgeUploadRequest, admin: dict = Depends(require_admin)) -> dict:
    try:
        chunk_count = await kb_store.upsert(req.title, req.content, admin["username"])
    except Exception as exc:
        logger.error("event=admin_upload_knowledge_fail", extra={"admin": admin["username"], "title": req.title, "error": str(exc)})
        raise HTTPException(status_code=502, detail="知识库同步失败，请稍后重试或点击「同步」按钮")
    await kb_store.reconcile_pending()
    logger.info(
        "event=admin_upload_knowledge",
        extra={"admin": admin["username"], "title": req.title, "chunks": chunk_count},
    )
    return {"count": chunk_count}


@router.get("/admin/knowledge")
async def list_knowledge(admin: dict = Depends(require_admin)) -> dict:
    items = await kb_store.list_docs()
    logger.info("event=admin_list_knowledge", extra={"admin": admin["username"], "count": len(items)})
    return {"items": items, "total": len(items)}


# 注意: sync 必须定义在 /{source} 路由之前，否则 "sync" 会被 {source} 路径参数捕获
@router.post("/admin/knowledge/sync")
async def sync_knowledge(force: bool = Query(False), admin: dict = Depends(require_admin)) -> dict:
    """全量对账：从 MySQL 重建全部文档 chunks + 清理孤儿。异常恢复时手动触发。

    force=true（?force=true）强制全量重建（升级 embedding/切分逻辑、清库恢复后）；
    默认增量跳检：内容未变且向量库在的文档跳过重建（content_hash 判据）。
    """
    result = await kb_store.sync_full(force=force)
    logger.info("event=admin_sync_knowledge", extra={"admin": admin["username"], **result})
    return result


@router.post("/admin/knowledge/upload", status_code=201)
async def upload_knowledge_file(
    file: UploadFile = File(...),
    title: str | None = Form(None),  # 可选：显式标题作 source，否则用文件名 stem
    admin: dict = Depends(require_admin),
) -> dict:
    """上传/覆盖一篇知识库文档（multipart 文件，仅 .md/.txt）。

    落库复用 kb_store.upsert：同 source 再次上传即覆盖更新，同内容幂等跳过，
    一致性（pending 补偿 / 全量对账）天然继承。
    """
    raw = await _read_limited(file, KB_UPLOAD_MAX_BYTES)
    source, content = _parse_knowledge_upload(file.filename or "", title or "", raw)
    try:
        chunk_count = await kb_store.upsert(source, content, admin["username"])
    except Exception as exc:
        logger.error("event=admin_upload_file_fail", extra={"admin": admin["username"], "source": source, "error": str(exc)})
        raise HTTPException(status_code=502, detail="知识库同步失败，请稍后重试或点击「同步」按钮")
    await kb_store.reconcile_pending()
    logger.info("event=admin_upload_file", extra={"admin": admin["username"], "source": source, "chunks": chunk_count})
    return {"source": source, "count": chunk_count}


@router.put("/admin/knowledge/{source}")
async def update_knowledge(source: str, req: KnowledgeUpdateRequest, admin: dict = Depends(require_admin)) -> dict:
    try:
        chunk_count = await kb_store.update_source(source, req.content, admin["username"])
    except Exception as exc:
        logger.error("event=admin_update_knowledge_fail", extra={"admin": admin["username"], "source": source, "error": str(exc)})
        raise HTTPException(status_code=502, detail="知识库同步失败，请稍后重试或点击「同步」按钮")
    await kb_store.reconcile_pending()
    logger.info(
        "event=admin_update_knowledge",
        extra={"admin": admin["username"], "source": source, "chunks": chunk_count},
    )
    return {"msg": "已更新", "chunks": chunk_count}


@router.delete("/admin/knowledge/{source}")
async def delete_knowledge(source: str, admin: dict = Depends(require_admin)) -> dict:
    await kb_store.delete_source(source, admin["username"])
    await kb_store.reconcile_pending()
    logger.info("event=admin_delete_knowledge", extra={"admin": admin["username"], "source": source})
    return {"msg": "已删除"}


# =============================================================
# Admin 订单管理（方案 9 章 API；种子数据维护）
# =============================================================
class AdminOrderItemIn(BaseModel):
    item_id: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)
    price: float = Field(ge=0)
    quantity: int = Field(default=1, ge=1)
    returnable: bool = True


class AdminOrderCreate(BaseModel):
    order_id: str = Field(min_length=1, max_length=32)
    user_id: int
    status: str = "PAID"
    total_amount: float = Field(ge=0)
    shipping_address: str = ""
    items: list[AdminOrderItemIn] = []


class AdminOrderUpdate(BaseModel):
    status: str | None = None
    total_amount: float | None = None
    shipping_address: str | None = None


@router.get("/admin/orders")
async def admin_list_orders(admin: dict = Depends(require_admin)) -> dict:
    rows = await mysql_pool.fetchall("SELECT * FROM orders ORDER BY id DESC")
    for row in rows:
        row["items"] = await mysql_pool.fetchall(
            "SELECT * FROM order_items WHERE order_id=%s ORDER BY id", (row["id"],)
        )
    logger.info("event=admin_list_orders", extra={"admin": admin["username"], "count": len(rows)})
    return {"items": rows}


@router.post("/admin/orders", status_code=201)
async def admin_create_order(req: AdminOrderCreate, admin: dict = Depends(require_admin)) -> dict:
    exist = await mysql_pool.fetchone("SELECT id FROM orders WHERE order_id=%s", (req.order_id,))
    if exist:
        raise HTTPException(status_code=409, detail="订单号已存在")
    async with mysql_pool.transaction() as run:
        cur = await run(
            "INSERT INTO orders (order_id, user_id, status, total_amount, shipping_address) VALUES (%s,%s,%s,%s,%s)",
            (req.order_id, req.user_id, req.status, req.total_amount, req.shipping_address),
        )
        order_db_id = cur.lastrowid
        for it in req.items:
            await run(
                "INSERT INTO order_items (order_id, item_id, name, price, quantity, returnable) VALUES (%s,%s,%s,%s,%s,%s)",
                (order_db_id, it.item_id, it.name, it.price, it.quantity, 1 if it.returnable else 0),
            )
    logger.info("event=admin_create_order", extra={"admin": admin["username"], "order_id": req.order_id})
    return {"msg": "已创建", "order_id": req.order_id}


@router.put("/admin/orders/{order_id}")
async def admin_update_order(order_id: str, req: AdminOrderUpdate, admin: dict = Depends(require_admin)) -> dict:
    exist = await mysql_pool.fetchone("SELECT id FROM orders WHERE order_id=%s", (order_id,))
    if not exist:
        raise HTTPException(status_code=404, detail="订单不存在")
    fields: list[str] = []
    params: list = []
    if req.status is not None:
        fields.append("status=%s")
        params.append(req.status)
    if req.total_amount is not None:
        fields.append("total_amount=%s")
        params.append(req.total_amount)
    if req.shipping_address is not None:
        fields.append("shipping_address=%s")
        params.append(req.shipping_address)
    if fields:
        params.append(order_id)
        await mysql_pool.execute(
            f"UPDATE orders SET {', '.join(fields)} WHERE order_id=%s", params
        )
    logger.info("event=admin_update_order", extra={"admin": admin["username"], "order_id": order_id})
    return {"msg": "已更新"}


@router.delete("/admin/orders/{order_id}")
async def admin_delete_order(order_id: str, admin: dict = Depends(require_admin)) -> dict:
    exist = await mysql_pool.fetchone("SELECT id FROM orders WHERE order_id=%s", (order_id,))
    if not exist:
        raise HTTPException(status_code=404, detail="订单不存在")
    async with mysql_pool.transaction() as run:
        await run("DELETE FROM order_items WHERE order_id=%s", (exist["id"],))
        await run("DELETE FROM orders WHERE id=%s", (exist["id"],))
    logger.info("event=admin_delete_order", extra={"admin": admin["username"], "order_id": order_id})
    return {"msg": "已删除"}


@router.post("/admin/orders/{order_id}/items", status_code=201)
async def admin_add_order_item(order_id: str, item: AdminOrderItemIn, admin: dict = Depends(require_admin)) -> dict:
    exist = await mysql_pool.fetchone("SELECT id FROM orders WHERE order_id=%s", (order_id,))
    if not exist:
        raise HTTPException(status_code=404, detail="订单不存在")
    await mysql_pool.execute(
        "INSERT INTO order_items (order_id, item_id, name, price, quantity, returnable) VALUES (%s,%s,%s,%s,%s,%s)",
        (exist["id"], item.item_id, item.name, item.price, item.quantity, 1 if item.returnable else 0),
    )
    logger.info("event=admin_add_order_item", extra={"admin": admin["username"], "order_id": order_id, "item_id": item.item_id})
    return {"msg": "已添加"}


@router.delete("/admin/orders/{order_id}/items/{item_db_id}")
async def admin_delete_order_item(order_id: str, item_db_id: int, admin: dict = Depends(require_admin)) -> dict:
    await mysql_pool.execute("DELETE FROM order_items WHERE id=%s", (item_db_id,))
    logger.info("event=admin_delete_order_item", extra={"admin": admin["username"], "order_id": order_id, "item_db_id": item_db_id})
    return {"msg": "已删除"}


# =============================================================
# Admin 测试数据一键重置（测试前恢复初始种子状态）
# =============================================================
# 种子订单/商品与 backend/sql/init.sql 保持一致。测试/验证会"消费"种子订单
#（退货后 SKU 变 RETURNED、订单被改状态等），重置即删光重插，保证"订单又都可以用"。
_SEED_ORDERS = [
    {"order_id": "ORD-20240801-001", "user_id": 2, "status": "DELIVERED", "total_amount": "69.70", "shipping_address": "上海市浦东新区示例路1号", "created_at": "2026-08-03 10:00:00", "delivered_at": "2026-08-03 15:00:00"},
    {"order_id": "ORD-20240805-002", "user_id": 2, "status": "SHIPPED", "total_amount": "228.90", "shipping_address": "上海市浦东新区示例路1号", "created_at": "2026-08-05 10:00:00", "delivered_at": None},
    {"order_id": "ORD-20240806-003", "user_id": 2, "status": "PAID", "total_amount": "89.85", "shipping_address": "上海市浦东新区示例路1号", "created_at": "2026-08-06 10:00:00", "delivered_at": None},
    {"order_id": "ORD-20240720-004", "user_id": 3, "status": "CANCELLED", "total_amount": "150.00", "shipping_address": "北京市朝阳区示例路2号", "created_at": "2026-07-20 10:00:00", "delivered_at": None},
    {"order_id": "ORD-20240725-005", "user_id": 2, "status": "DELIVERED", "total_amount": "88.00", "shipping_address": "上海市浦东新区示例路1号", "created_at": "2026-07-25 10:00:00", "delivered_at": "2026-07-25 16:00:00"},
]

# (order_id, item_id, name, price, quantity, returnable)
_SEED_ITEMS = [
    ("ORD-20240801-001", "SKU-001", "手机壳", "29.90", 1, 1),
    ("ORD-20240801-001", "SKU-002", "钢化膜", "19.90", 2, 1),
    ("ORD-20240805-002", "SKU-003", "蓝牙耳机", "199.00", 1, 1),
    ("ORD-20240805-002", "SKU-004", "耳机收纳盒", "29.90", 1, 1),
    ("ORD-20240806-003", "SKU-005", "数据线", "29.95", 2, 1),
    ("ORD-20240806-003", "SKU-006", "定制手机支架", "29.95", 1, 0),
    ("ORD-20240720-004", "SKU-007", "充电宝", "150.00", 1, 1),
    ("ORD-20240725-005", "SKU-008", "台灯", "88.00", 1, 1),
]


@router.post("/admin/reset-demo")
async def admin_reset_demo(admin: dict = Depends(require_admin)) -> dict:
    """测试数据一键重置：清空退货/退款/工单/商品/订单，恢复 init.sql 种子订单。

    用途：测试/验证脚本会消费种子订单（退货后 SKU 变 RETURNED、状态被改），
    测试前调用恢复到初始状态。仅 admin 可调用，事务内原子执行。
    """
    items_by_order: dict[str, list] = {}
    for it in _SEED_ITEMS:
        items_by_order.setdefault(it[0], []).append(it)

    async with mysql_pool.transaction() as run:
        await run("DELETE FROM return_orders")
        await run("DELETE FROM refund_orders")
        await run("DELETE FROM complaint_tickets")
        await run("DELETE FROM order_items")  # 先删子表，再删 orders（FK 约束）
        await run("DELETE FROM orders")
        for o in _SEED_ORDERS:
            cur = await run(
                "INSERT INTO orders (order_id, user_id, status, total_amount, shipping_address, created_at, delivered_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (o["order_id"], o["user_id"], o["status"], o["total_amount"], o["shipping_address"], o["created_at"], o["delivered_at"]),
            )
            oid = cur.lastrowid
            for order_id, item_id, name, price, quantity, returnable in items_by_order.get(o["order_id"], []):
                await run(
                    "INSERT INTO order_items (order_id, item_id, name, price, quantity, returnable) VALUES (%s,%s,%s,%s,%s,%s)",
                    (oid, item_id, name, price, quantity, returnable),
                )
    logger.info("event=admin_reset_demo", extra={"admin": admin["username"], "orders": len(_SEED_ORDERS)})
    return {"msg": "测试数据已重置", "orders": len(_SEED_ORDERS)}
