"""意图判定规则层：高置信模板化表达跳过 LLM 分类。

classify_intent() 开头短路调用 match_intent_rules()：命中返回 RuleHit（与 LLM 输出同构），
未命中返回 None 回退 LLM。只接管正则可精确锁定的模式，政策咨询（POLICY_INQUIRY）刻意不接管。

顺序求值（顺序即语义，第一条命中即返回）：
1. 纯问候（整句锚定，先于疑问门——"在吗" 含"吗"）
2. 裸查单（整句锚定的订单查询句式，先于疑问门——"订单什么时候到"）
3. 疑问词门：疑问句式=政策/资格咨询语义，交给 LLM（"能退货吗" vs "我要退货"）
4. 动作规则：退货 / 退款 / 投诉（前缀紧邻约束，防 "帮我查一下退货政策" 误判）
5. 订单号 + 状态语义词 → ORDER_STATUS（放动作之后，防 "ORD-xxx 物流太慢，我要投诉" 被抢）
6. 兜底 None

不 import intent.py（避免循环依赖），只定义 RuleHit 与匹配函数。
"""
import re
from dataclasses import dataclass, field

# 订单号：ORD-20240801-001 / ORD20240801001（允许中段与尾段数字）
ORDER_ID_RE = re.compile(r"[Oo][Rr][Dd][-]?\d{6,}(?:-\d{1,4})?")

# 纯问候（整句锚定，忽略大小写；混合句如"你好，我要退货"不匹配，落到动作规则）
_GREETING_RE = re.compile(
    r"^(?:你好|您好|hi|hello|哈喽|嗨|在吗|你好呀|您好呀|早上好|中午好|下午好|晚上好)$",
    re.IGNORECASE,
)

# 裸查单：整句即订单状态查询（先于疑问门，覆盖"订单什么时候到"类疑问句式）
_BARE_ORDER_QUERY_RE = re.compile(
    r"^(?:查订单|查单|查询订单|我的订单|查一下订单|看看订单|看订单"
    r"|订单什么时候到|订单到哪了|订单到没到|订单发货没|订单状态"
    r"|物流到哪了|物流状态|发货没)$"
)

# 疑问词门：命中即视为政策/资格咨询，回退 LLM
_QUESTION_RE = re.compile(
    r"(吗|能不能|可不可以|行不行|是否|怎么|如何|多久|多长时间|为什么|啥时候|什么时候|几点)"
)

# 退货动作：动作前缀必须紧邻"退货"，防"帮我查一下退货政策"误判；部分退货"只退xx"单独提取
_RETURN_RE = re.compile(r"(?:我要|我想|申请|帮我|准备)(?:退货|退回|退掉)|^退货$")
_RETURN_ITEMS_RE = re.compile(r"(?:只|就)退(.+?)(?:吧|了|就行|就够了|，|,|。|！|!|$)")
# 退款动作（不提取 items，与 LLM 语义一致）
_REFUND_RE = re.compile(r"(?:我要|我想|申请|帮我|准备)(?:退款|退钱|仅退款)|^(?:退款|仅退款)$")
# 投诉动作
_COMPLAINT_RE = re.compile(r"(?:我要|我想|现在|就要|必须|一定要)(?:投诉|举报)|^(?:投诉|举报)$")

# 订单号存在时，出现状态查询语义词即判 ORDER_STATUS（放在动作规则之后）
_ORDER_STATUS_SEMANTIC_RE = re.compile(r"(到哪|状态|物流|进度|发货|快递|什么时候到|到没到)")


@dataclass(frozen=True)
class RuleHit:
    """规则命中结果，与 LLM 分类输出同构（intent/slots/missing_slots/summary）。"""

    intent: str
    slots: dict = field(default_factory=dict)
    missing_slots: list = field(default_factory=list)
    summary: str = ""


def _strip_tail(text: str) -> str:
    """去掉句末标点/空白，使整句锚定正则可命中"你好！""订单什么时候到？"等。"""
    return re.sub(r"[？?。！!，,、\s]+$", "", text.strip())


def _order_id_of(text: str) -> str | None:
    m = ORDER_ID_RE.search(text)
    return m.group(0) if m else None


def _partial_return_items(text: str) -> list[dict] | None:
    """部分退货 items 提取（"只退手机壳"），琐碎捕获（款/钱/掉/货）视为噪声丢弃。"""
    m = _RETURN_ITEMS_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    if not name or name in {"款", "钱", "掉", "货"}:
        return None
    return [{"name": name}]


def match_intent_rules(text: str) -> RuleHit | None:
    """规则匹配入口：命中返回 RuleHit，未命中返回 None（交给 LLM 分类）。"""
    text = _strip_tail(text)
    if not text:
        return None

    # 1. 纯问候
    if _GREETING_RE.match(text):
        return RuleHit(intent="CHITCHAT", summary="问候")

    # 2. 裸查单（整句订单查询）
    if _BARE_ORDER_QUERY_RE.match(text):
        return RuleHit(intent="ORDER_STATUS", missing_slots=["order_id"], summary="查询订单状态")

    # 3. 疑问词门：疑问句式回退 LLM
    if _QUESTION_RE.search(text):
        return None

    # 4. 动作规则：退货 / 退款 / 投诉
    if _RETURN_RE.search(text):
        oid = _order_id_of(text)
        slots: dict = {"order_id": oid} if oid else {}
        items = _partial_return_items(text)
        if items:
            slots["items"] = items
        return RuleHit(
            intent="RETURN_REQUEST",
            slots=slots,
            missing_slots=[] if oid else ["order_id"],
            summary=f"申请退货{(' ' + oid) if oid else ''}",
        )
    if _REFUND_RE.search(text):
        oid = _order_id_of(text)
        return RuleHit(
            intent="REFUND_REQUEST",
            slots={"order_id": oid} if oid else {},
            missing_slots=[] if oid else ["order_id"],
            summary=f"申请退款{(' ' + oid) if oid else ''}",
        )
    if _COMPLAINT_RE.search(text):
        oid = _order_id_of(text)
        return RuleHit(
            intent="COMPLAINT",
            slots={"order_id": oid} if oid else {},
            missing_slots=[],
            summary="用户投诉",
        )

    # 5. 订单号 + 状态语义词 → 订单状态查询
    oid = _order_id_of(text)
    if oid and _ORDER_STATUS_SEMANTIC_RE.search(text):
        return RuleHit(
            intent="ORDER_STATUS",
            slots={"order_id": oid},
            missing_slots=[],
            summary=f"查询订单 {oid} 状态",
        )

    return None
