"""FastAPI 入口。

- lifespan: 启动时初始化 Redis 会话 + MySQL 连接池；关闭时优雅回收。
- 路由挂载: /api/v1/auth/* 认证, /api/v1/* 业务。
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import auth, routes
from app.infrastructure.deepseek import deepseek_client
from app.infrastructure.mysql import mysql_pool
from app.rag.init import init_knowledge
from app.rag.retriever import retriever
from app.session.manager import session_manager
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("event=app_startup")
    await session_manager.init()
    await mysql_pool.init()
    await init_knowledge()
    await retriever.init()
    await deepseek_client.init()
    yield
    # ---- 优雅关闭 ----
    logger.info("event=shutdown_start 停止接受新请求，等待活跃请求完成")
    # 活跃会话已实时双写 MySQL（StorageRouter.save），无需额外 checkpoint
    await session_manager.close()
    await mysql_pool.close()
    await retriever.close()
    await deepseek_client.close()
    logger.info("event=shutdown_done")


app = FastAPI(title="AI 智能客服", version="0.1.0", lifespan=lifespan)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(routes.router, prefix="/api/v1")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
