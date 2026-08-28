"""FastAPI 入口。

- lifespan: 启动时初始化 Redis 会话 + MySQL 连接池；关闭时优雅回收。
- 路由挂载: /api/v1/auth/* 认证, /api/v1/* 业务。
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from app.api import auth, contracts, routes
from app.config import settings, validate_security_config
from app.infrastructure import metrics
from app.infrastructure.deepseek import deepseek_client
from app.infrastructure.mysql import mysql_pool
from app.infrastructure.schema import init_schema
from app.rag.init import init_knowledge
from app.rag.retriever import retriever
from app.session.cleaner import session_cleaner
from app.session.manager import session_manager
from app.utils.logger import logger
from app.utils.trace import TraceIdFilter, trace_id_var

# 链路追踪：trace_id 注入 JSON 日志（JsonFormatter 自动合并非保留字段，formatter 零改动）
logger.addFilter(TraceIdFilter())


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_security_config()  # fail-fast：弱 JWT 密钥 / 弱 admin 口令拒绝启动
    if settings.allow_weak_admin_password and settings.app_env != "prod":
        logger.warning("ALLOW_WEAK_ADMIN_PASSWORD=true：admin 弱口令逃生开关已开启，仅限演示环境（APP_ENV=dev）")
    logger.info("event=app_startup")
    await session_manager.init()
    await mysql_pool.init()
    await init_schema()   # 共享 mysql 不跑 init.sql，应用启动自建表+种子（幂等）
    await init_knowledge()
    await retriever.init()
    await deepseek_client.init()
    session_cleaner.start()  # TTL 清理：依赖 mysql_pool 已就绪
    yield
    # ---- 优雅关闭 ----
    logger.info("event=shutdown_start 停止接受新请求，等待活跃请求完成")
    # 活跃会话已实时双写 MySQL（StorageRouter.save），无需额外 checkpoint
    await session_cleaner.stop()
    await session_manager.close()
    await mysql_pool.close()
    await retriever.close()
    await deepseek_client.close()
    logger.info("event=shutdown_done")


app = FastAPI(title="AI 智能客服", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """链路追踪：取网关透传的 X-Request-ID（无则生成 uuid），写入 contextvar 供日志
    filter 使用，并在响应头回传（经网关时网关会隐藏后端重复头，无副作用）。"""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    trace_id_var.set(rid)
    response = await call_next(request)
    response.headers.setdefault("X-Request-ID", rid)
    return response


app.include_router(auth.router, prefix="/api/v1")
app.include_router(routes.router, prefix="/api/v1")
app.include_router(contracts.router, prefix="/api")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus 文本格式指标：LLM 调用量/失败率/熔断/排队、会话锁等待、意图规则命中率。"""
    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")
