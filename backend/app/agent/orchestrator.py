"""Agent 编排器（完整版 6 阶段流水线）。

INPUT → [1.预处理] → [2.意图识别+切换] → [3.上下文装配] → [4.状态推进] → [5.动作执行] → [6.响应生成] → OUTPUT

- 有进行中的业务状态机时，分类结果判断"推进"还是"切换"（保存快照）
- CHITCHAT 前 2 轮 LLM 自由回复，第 3 轮收束，第 4 轮规则话术
- POLICY_INQUIRY 走 RAG 检索注入
"""
import re
import time
from functools import wraps

from app.agent.intent import IntentResult, classify_intent
from app.agent.rule_engine import match_rule
from app.agent.state_machine.complaint_flow import ComplaintFlow
from app.agent.state_machine.refund_flow import RefundFlow
from app.agent.state_machine.return_flow import ReturnFlow
from app.config import settings
from app.infrastructure.deepseek import (
    AllKeysDownError,
    CapacityExceededError,
    LLMUnavailableError,
    deepseek_client,
)
from app.services import order_service
from app.session.models import Session
from app.utils.logger import logger

LLM_FALLBACK_ERRORS = (LLMUnavailableError, CapacityExceededError, AllKeysDownError)

INJECTION_RE = re.compile(
    r"忽略(之前|以上)?(的)?(所有|全部)?(指令|规则|提示)|无视(指令|规则)|"
    r"system\s*prompt|ignore\s+(all\s+)?previous|绕过|越狱",
    re.I,
)

FLOWS = {
    "RETURN_REQUEST": ReturnFlow(),
    "REFUND_REQUEST": RefundFlow(),
    "COMPLAINT": ComplaintFlow(),
}

STATUS_DESC = {"PAID": "已付款待发货", "SHIPPED": "已发货运输中", "DELIVERED": "已签收", "CANCELLED": "已取消"}
SWITCH_THRESHOLD = 0.8


def _detect_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text))


# 部分退货规则兜底：LLM 未提取 items 槽时，从"只退/就退/只要退/仅退/单退 + 商品名"句式降级提取。
# 只匹配强指定信号词；捕获后再精化，避免把普通原因/全额语义/否定句式误当商品名：
# - 前缀"了/个"为完成时语气词与量词（"只退了个手机壳"→"手机壳"）
# - 后缀 strip 语气/量词尾缀（"只退手机壳吧"→"手机壳"、"只退手机壳一个"→"手机壳"）
# - 多商品按"和/、/斜杠"拆分（"只退手机壳和钢化膜"→两个商品）
# - 停用词过滤"款/货/单/运费/差价"等，拦截"只退款""就退货吧"等全额语义
# - 否定句式（"不要只退手机壳"/"只退手机壳不行"）与用户意图相反，整体跳过
_PARTIAL_RETURN_RE = re.compile(r"(?:只退|就退|只要退|仅退|单退)\s*([^\s，,。.!！?？]{1,16})")
_PARTIAL_RETURN_PREFIX_RE = re.compile(r"^(?:了|个)+")
_PARTIAL_RETURN_TRAIL_RE = re.compile(r"(?:一个|[吧啊呀哈哦了呢么的下])+$")
_PARTIAL_RETURN_SPLIT_RE = re.compile(r"[和、/／]")
_PARTIAL_RETURN_NOISE = {"款", "货", "单", "运费", "差价", "费用", "钱"}
_PARTIAL_RETURN_NEG_RE = re.compile(
    r"(?:不要|别|不想|不能|不用)\s*(?:只退|就退|只要退|仅退|单退)"
    r"|(?:只退|就退|只要退|仅退|单退)[^，,。.!！?？]{0,16}(?:不行|不要|算了)"
)


def _extract_partial_items(text: str) -> list:
    """从用户输入兜底提取指定退货商品名；无匹配或语义为否定/全额时返回空列表。"""
    if _PARTIAL_RETURN_NEG_RE.search(text):
        return []
    items = []
    for m in _PARTIAL_RETURN_RE.finditer(text):
        name = _PARTIAL_RETURN_PREFIX_RE.sub("", m.group(1))
        name = _PARTIAL_RETURN_TRAIL_RE.sub("", name).strip()
        for piece in _PARTIAL_RETURN_SPLIT_RE.split(name):
            piece = piece.strip()
            if piece and piece not in _PARTIAL_RETURN_NOISE and piece not in items:
                items.append(piece)
    return items


def _rule_engine_fallback(fn):
    """LLM 全部不可用（熔断）时降级规则引擎。"""

    @wraps(fn)
    async def wrapper(session, user_message, user_id, emit=None):
        try:
            return await fn(session, user_message, user_id, emit)
        except LLM_FALLBACK_ERRORS:
            logger.warning(
                "event=rule_engine_triggered",
                extra={"session_id": session.session_id, "user_id": user_id},
            )
            return match_rule(user_message)

    return wrapper


def _state_context(session: Session) -> str | None:
    """状态机上下文，注入意图分类 prompt，防止业务流中的"确认"误判 CHITCHAT。"""
    if session.intent and session.agent_state and session.intent in FLOWS:
        stage = session.agent_state.get("stage", "")
        return f"用户正在办理 {session.intent}，当前节点 {stage}。若输入是流程推进（确认/补充信息/取消），归为该意图而非 CHITCHAT。"
    return None


def _init_state(intent: str, result: IntentResult, session: Session, user_id: int) -> dict:
    slots = result.slots or {}
    sid = session.session_id
    if intent == "RETURN_REQUEST":
        oid = slots.get("order_id")
        # return_items: 用户指定只退的部分商品（商品名数组），空列表=退全部可退商品
        return {"user_id": user_id, "session_id": sid, "order_id": oid,
                "return_items": slots.get("items", []) or [],
                "stage": "verify_order" if oid else "collect_order_id"}
    if intent == "REFUND_REQUEST":
        oid = slots.get("order_id")
        return {"user_id": user_id, "session_id": sid, "order_id": oid,
                "stage": "verify_order" if oid else "collect_order_id"}
    if intent == "COMPLAINT":
        return {"user_id": user_id, "session_id": sid,
                "complaint_type": slots.get("complaint_type"), "stage": "collect_complaint_type"}
    return {}


async def _handle_order_status(user_id: int, slots: dict) -> str:
    order_id = slots.get("order_id")
    if not order_id:
        # 无订单号时列出最近订单辅助定位；空列表友好提示
        orders = await order_service.list_user_orders(user_id, limit=5)
        if not orders:
            return "您最近没有订单，请直接告诉我订单号。"
        lines = "、".join(f"{o.order_id}（{STATUS_DESC.get(o.status, o.status)}）" for o in orders)
        return f"请提供订单号。您最近的订单：{lines}。"
    order = await order_service.query_order(order_id, user_id)
    if not order:
        return "未找到该订单，请核对订单号。也可以让我列出您最近的订单。"
    items = "、".join(f"{i.name}×{i.quantity}" for i in order.items)
    return (
        f"订单 {order.order_id} 当前状态：{STATUS_DESC.get(order.status, order.status)}。"
        f"商品：{items}；金额：¥{order.total_amount}。"
    )


async def _handle_policy(user_message: str) -> str:
    from app.rag.retriever import retriever

    results = await retriever.search(user_message)
    if not results:
        return "您的问题暂未收录到知识库，建议联系人工客服确认（热线 400-XXX-XXXX）。"
    ctx = "\n\n".join(f"[{r.metadata.get('source', '')}] {r.text}" for r in results)
    sys = (
        "你是电商客服，基于以下政策文档回答用户问题。只依据文档内容回答，"
        "文档未覆盖的请说明需人工确认。\n\n【政策文档】\n" + ctx
    )
    data = await deepseek_client.chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user_message}]
    )
    return data["choices"][0]["message"]["content"].strip()


async def _handle_chitchat(session: Session, user_message: str, user_id: int) -> str:
    rounds = (session.agent_state or {}).get("chitchat_round", 0)
    if rounds >= 3:  # 第 4 轮起规则话术
        return "我是智能客服，专注于订单查询、退换货、退款和投诉处理。需要帮助请直接告诉我订单号或问题哦。"
    if rounds >= 2:  # 第 3 轮温和收束
        sys = "你是智能客服。请简短友好回应，并温和地把话题引导回订单/售后业务。"
    else:
        sys = "你是智能客服，请用简体中文简短友好地回复用户。"
    data = await deepseek_client.chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user_message}]
    )
    return data["choices"][0]["message"]["content"].strip()


@_rule_engine_fallback
async def run_agent(session: Session, user_message: str, user_id: int, emit=None) -> str:
    """单轮对话：6 阶段流水线。emit 为可选的 SSE 事件回调。"""
    t0 = time.perf_counter()

    async def status(stage: str, msg: str) -> None:
        if emit:
            await emit({"type": "status", "stage": stage, "message": msg})

    # ===== 阶段 1: 预处理（注入检测）=====
    await status("preprocess", "正在处理您的问题...")
    if _detect_injection(user_message):
        logger.warning("event=injection_detected", extra={"session_id": session.session_id, "user_id": user_id})
        return "检测到可能的异常输入，为了安全已忽略。请正常描述您的问题。"

    # ===== 阶段 2: 意图识别 + 切换判断 =====
    await status("intent", "正在理解您的问题...")
    in_business_flow = session.intent in FLOWS and bool(session.agent_state)

    if in_business_flow:
        # 进行中的业务状态机：分类结果判断推进还是切换
        intent_result = await classify_intent(user_message, _state_context(session))
        if intent_result.intent == session.intent or intent_result.confidence < SWITCH_THRESHOLD:
            intent = session.intent  # 推进当前状态机
        else:
            # 切换意图：保存快照，开始新流程
            session.snapshots[session.intent] = session.agent_state
            session.agent_state = None
            session.intent = None
            intent = intent_result.intent
            logger.info("event=intent_switch", extra={"session_id": session.session_id,
                        "from": intent_result.intent, "saved": session.snapshots.keys()})
    else:
        intent_result = await classify_intent(user_message, _state_context(session))
        intent = intent_result.intent
    logger.info("event=stage_intent", extra={"session_id": session.session_id, "intent": intent})

    # ===== 阶段 3+4+5: 上下文装配 + 状态推进 + 动作执行 =====
    reply = ""

    if intent in FLOWS:
        # 有快照 → 恢复（用户回到该意图）
        if not session.agent_state and session.snapshots.get(intent):
            session.agent_state = session.snapshots.pop(intent)
            session.intent = intent
            logger.info("event=snapshot_restored", extra={"session_id": session.session_id, "intent": intent})
        if not session.agent_state:
            session.agent_state = _init_state(intent, intent_result, session, user_id)
            await status("order_query", "正在为您办理...")
        else:
            await status("order_query", "正在处理...")
        # 多轮补充指定退货商品：LLM 在任意推进轮都可能提取 items 槽。若仅 _init_state 注入，
        # 后轮（如先"我要退货 ORD-001"再"只退手机壳"）会被丢弃且被误当退货原因、按全量确认。
        # 此处合并本轮 items（LLM 未提取时用"只退/就退/只要退 + 商品名"句式规则兜底）；
        # 资格已算过（已过 check_eligibility）则回到该节点携带新子集重算。
        if intent == "RETURN_REQUEST" and session.agent_state:
            items = (intent_result.slots or {}).get("items") or _extract_partial_items(user_message)
            if items:
                merged = {**session.agent_state, "return_items": items}
                if merged.get("stage") in ("collect_reason", "confirm"):
                    merged.update({"stage": "check_eligibility", "eligibility": None, "awaiting": None})
                session.agent_state = merged
        new_state = await FLOWS[intent].step(session.agent_state, user_message)
        session.agent_state = new_state
        session.intent = intent
        reply = new_state.get("message", "")
        # 到达确认节点 → 通知前端渲染 ConfirmButton。
        # 注意必须先排除 final：取消时 _confirm 返回 final 但 awaiting 残留旧值 "confirm"，
        # 若不排除会在取消响应里误发 confirm action，导致前端确认按钮残留。
        if not new_state.get("final") and new_state.get("awaiting") == "confirm" and emit:
            await emit({"type": "action", "action": "confirm", "intent": intent})
        if new_state.get("final"):
            session.agent_state = None
            session.intent = None

    elif intent == "ORDER_STATUS":
        await status("order_query", "正在查询订单...")
        reply = await _handle_order_status(user_id, intent_result.slots)

    elif intent == "POLICY_INQUIRY":
        await status("rag", "正在检索政策...")
        try:
            reply = await _handle_policy(user_message)
        except LLM_FALLBACK_ERRORS:
            raise  # 熔断 → 传播给装饰器 → 规则引擎
        except Exception:
            reply = "系统繁忙，请稍后再试。"
        session.agent_state = None
        session.intent = None

    else:  # CHITCHAT
        reply = await _handle_chitchat(session, user_message, user_id)
        session.agent_state = {"chitchat_round": (session.agent_state or {}).get("chitchat_round", 0) + 1}
        session.intent = intent

    # ===== 阶段 6: 响应生成 =====
    logger.info(
        "event=request_out",
        extra={"session_id": session.session_id, "intent": intent,
               "ms": round((time.perf_counter() - t0) * 1000)},
    )
    return reply
