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


def _obs():
    """惰性取 obs_sdk 模块：未启用（config 三要素缺一）或未安装时返回 None。

    观测边带不阻塞业务（§11.3 接入）：装配是可选增强，配置缺失/包缺失都让服务照常跑。
    放模块级而非 lifespan——request 中间件（http 层，与 lifespan 无关）也要每请求判观测。
    """
    if not settings.obs_ready:
        return None
    try:
        import obs_sdk
    except ImportError:
        logger.warning("[obs] obs_sdk 未安装，观测边带关闭（OBS_ENABLED=true 但包缺失）")
        return None
    return obs_sdk


def _obs_end(obs, response, request, aborted: bool = False) -> None:
    """request 出口统一收口：断连 > HTTP 状态码 > ok。

    SSE 业务层断连（routes.send_message 断连处置已置 request.state.obs_aborted）与 body
    迭代异常（客户端中途关闭未走业务层）都归 error + CLIENT_DISCONNECT——trace 如实反映
    「未拿到完整响应」；非 2xx 按 HTTP_xxx 记 error；其余 ok。end_request 自判 status
    合法性/补 duration，此处不重复。

    入参由业务路由置 request.state.obs_input（同 obs_aborted 惯例），三条出口都带上——
    error 路径同样需要现场，否则失败 trace 建不出簇。
    """
    obs_input = getattr(request.state, "obs_input", None)
    if aborted or getattr(request.state, "obs_aborted", False):
        obs.end_request("error", error_type="CLIENT_DISCONNECT", error_msg="客户端连接中断",
                        input=obs_input)
        return
    code = response.status_code
    if code >= 400:
        obs.end_request("error", error_type=f"HTTP_{code}", input=obs_input)
        return
    obs.end_request("ok", input=obs_input)


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
    # obs_sdk 装配（观测边带，§11.3 cs #1）：cs 业务 logger "cs" propagate=False → root
    # handler 收不到，须 extra_loggers=["cs"] 点名（sdk>=0.1.1）。init 失败仅告警不拦启动。
    obs = _obs()
    if obs is not None:
        try:
            obs.init(
                "customer-service",
                kafka_servers=settings.obs_kafka_servers,
                topic=settings.obs_kafka_topic,
                sasl_username=settings.obs_kafka_sasl_username or None,
                sasl_password=settings.obs_kafka_sasl_password or None,
                flush_batch=settings.obs_flush_batch,
                flush_interval_s=settings.obs_flush_interval_s,
                log_mode="stdlib",
                extra_loggers=["cs"],
            )
            logger.info("[obs] obs_sdk 已初始化 topic=%s", settings.obs_kafka_topic)
        except Exception as e:  # 观测边带故障不拦服务启动
            logger.warning("[obs] obs_sdk init 失败（观测边带关闭）: %s", e)
    yield
    # ---- 优雅关闭 ----
    logger.info("event=shutdown_start 停止接受新请求，等待活跃请求完成")
    # 活跃会话已实时双写 MySQL（StorageRouter.save），无需额外 checkpoint
    await session_cleaner.stop()
    await session_manager.close()
    await mysql_pool.close()
    await retriever.close()
    await deepseek_client.close()
    if obs is not None:
        try:
            obs.shutdown()  # 终刷剩余事件后关线程（幂等：未 init 也安全）
        except Exception as e:
            logger.warning("[obs] obs_sdk shutdown 异常: %s", e)
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


@app.middleware("http")
async def obs_request_middleware(request: Request, call_next):
    """观测 request 入口/出口（§11.3 cs #2）：SSE 在 body 迭代完成后聚合打一条。

    - begin_request：method/path 交给 SDK 归一 interface（动态段 → {id}）；trace_id 沿用
      网关透传 X-Request-ID（trace_middleware 同源，无则 SDK 自造）。
    - end_request 时点：StreamingResponse 的 body 迭代完成才收口——SSE 的 LLM 调用发生在
      response body 发送期（call_next 返回时尚未开始），若即时 end 则 request duration≈0
      且锚点(seq=0)晚于子节点事件。包一层透传迭代器，真实流式不缓冲。
    - 客户端断连：routes.send_message 业务层断连处置置 request.state.obs_aborted → 记
      error+CLIENT_DISCONNECT；body 迭代异常（上传中断等）同记。未启用时零开销直通。
    """
    obs = _obs()
    if obs is None:
        return await call_next(request)

    obs.begin_request(method=request.method, path=request.url.path,
                      trace_id=request.headers.get("X-Request-ID"))
    try:
        response = await call_next(request)
    except Exception:
        obs.end_request("error", error_type="UNHANDLED_EXCEPTION")
        raise

    body_iter = getattr(response, "body_iterator", None)
    if body_iter is None:
        # 非流式响应体已整体生成：直接收口（status_code 即可判定）
        _obs_end(obs, response, request)
        return response

    async def _body_with_obs():
        try:
            async for chunk in body_iter:
                yield chunk
        except BaseException:
            _obs_end(obs, response, request, aborted=True)
            raise
        else:
            _obs_end(obs, response, request)

    response.body_iterator = _body_with_obs()
    return response


app.include_router(auth.router, prefix="/api")  # T15：登录路由统一 /api/auth/login（与 gq/cc 一致）
app.include_router(routes.router, prefix="/api/v1")
app.include_router(contracts.router, prefix="/api")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus 文本格式指标：LLM 调用量/失败率/熔断/排队、会话锁等待、意图规则命中率。"""
    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")
