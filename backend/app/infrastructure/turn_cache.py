"""回合级 LLM 答案缓存（Redis 精确 key + 查询归一化）。

借鉴 good-question chat_cache 的取舍：整轮缓存 + 仅无状态轮次 + 查询归一化，
命中时短路整条 agent 图（零 LLM 调用）。明确不做语义缓存/独立规划缓存，
原因见 agent/orchestrator.py run_agent 的方案注释：
- 语义缓存：embedding 相似 ≠ 答案可复用（"退货政策" vs "退款政策"），
  且检索层省的是计算不是 LLM token；查询归一化已覆盖表面变体且零误命中。
- 独立规划缓存：POLICY 的决策已被整轮缓存覆盖，独立层冗余。

Redis 不可用/超时 → 静默降级（缓存是纯优化，绝不阻断主链路）；首个失败进入
_DISABLE_COOLDOWN 冷却，避免故障/测试环境下反复 1s 连接超时拖慢主链路。
"""
import hashlib
import json
import re
import time

from app.config import settings

_CACHE_VERSION = 2  # v2：search_policy 结果信封化（{ok,data,error}），旧 v1 裸 dict 缓存全部失效
_DISABLE_COOLDOWN = 60.0
_PREFIX = "cs:turn:"

# 全角 → 半角（ASCII 可见区）。U+3000 全角空格由下方 \s+ 折叠处理。
_FULLWIDTH_MAP = {
    ord(c): ord(c) - 0xFEE0
    for c in "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？"
             "＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀"
             "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～"
}
# 客套后缀（剥掉后语义不变，纯客套/语气词），前缀只保留「纯询问客套」：
# - "请问/我想问/想问/请告诉我" 是询问语气，剥掉后仍是政策咨询（"请问退货政策"→"退货政策"），安全；
# - 不剥 "帮我/麻烦/帮我查*" 等动作性前缀：它们带动作倾向，"帮我退货" 剥"帮我" 会归一成裸动作词
#   "退货"，与新会话业务流消息（本应进 RETURN/REFUND 状态机）撞缓存 key、被短路成政策答复。
#   customer-service 有业务流状态机，前缀剥离必须保守（good-question 无状态机故无此约束）。
_POLITE_PREFIX_RE = re.compile(r"^(请告诉我|我想问|想问|请问)")
_POLITE_SUFFIX_RE = re.compile(r"(谢谢你|麻烦你了|谢谢|呢|吧|啊|呀|哦|了)$")

_client = None
_last_fail = 0.0  # 最近一次 Redis 失败时刻（monotonic），冷却期内不再尝试


def _redis():
    """惰性 Redis 客户端（同 retriever 风格，import 不建连）。"""
    global _client
    if _client is None:
        import redis.asyncio as aioredis

        _client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
    return _client


def _disabled() -> bool:
    """冷却期：最近一次失败后 _DISABLE_COOLDOWN 秒内不再尝试连接。"""
    return time.monotonic() - _last_fail < _DISABLE_COOLDOWN


def _mark_fail() -> None:
    global _last_fail
    _last_fail = time.monotonic()


def normalize_query(text: str) -> str:
    """规则化清洗：全角→半角、折叠空白、剥客套前后缀。

    纯规则变换，零语义误命中；让"请问退货时限几天？"与"退货时限几天"
    落到同一缓存 key（对应 good-question _normalize_query 的思路）。
    """
    t = (text or "").strip().translate(_FULLWIDTH_MAP)
    t = re.sub(r"\s+", " ", t)
    # 尾部标点归一："退货几天？/。" 与 "退货几天" 同一 key（半角 ? 由全角映射得到）
    t = re.sub(r"[?？！!。.\s]+$", "", t)
    t = _POLITE_PREFIX_RE.sub("", t)
    # 前缀后常跟分隔符（"请问：退货时限" / "帮我，查一下"），剥掉避免残留冒号进 key
    t = re.sub(r"^[：:，,、。\s]+", "", t)
    t = _POLITE_SUFFIX_RE.sub("", t).strip()
    return t


def turn_key(normalized: str) -> str:
    """缓存 key：模型名进 key（换模型即失效），sha256 前 16 hex（同 good-question）。

    答案由 LLM 模型 + 检索 embedding 共同决定，任一变更旧答案即失效。
    """
    raw = f"{settings.deepseek_model_chat}|{settings.embedding_model}|{normalized}"
    return f"{_PREFIX}{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


async def get(key: str) -> dict | None:
    """取缓存；miss / 版本不符 / Redis 错误 / 冷却期 → None（不阻断主链路）。"""
    if _disabled():
        return None
    try:
        raw = await _redis().get(key)
    except Exception:
        _mark_fail()
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    # 防御非 dict payload（旧版本/误写进 Redis 的字符串、列表等）：get 会抛 AttributeError
    if not isinstance(payload, dict) or payload.get("v") != _CACHE_VERSION:
        return None
    return payload


async def set(key: str, payload: dict, ttl: int) -> None:
    """写缓存；失败静默（不影响主链路）。"""
    if _disabled():
        return
    try:
        await _redis().set(key, json.dumps(payload, ensure_ascii=False), ex=ttl)
    except Exception:
        _mark_fail()


async def flush_all() -> None:
    """清全部回合缓存（KB 变更后调用，防旧答案命中）。best-effort。"""
    if _disabled():
        return
    try:
        r = _redis()
        async for k in r.scan_iter(match=f"{_PREFIX}*", count=100):
            await r.delete(k)
    except Exception:
        _mark_fail()
