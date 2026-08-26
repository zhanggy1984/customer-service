"""回合缓存模块单测（FakeRedis，不依赖真实 Redis/模型）。"""
import pytest

from app.infrastructure import turn_cache as tc


class FakeRedis:
    """内存版 Redis：get/set/delete/scan_iter（decode_responses 语义，str key/value）。"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str):
        return self.data.get(key)

    async def set(self, key: str, value: str, ex=None) -> None:
        self.data[key] = value

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)

    async def scan_iter(self, match=None, count=100):
        prefix = match[:-1] if match and match.endswith("*") else (match or "")
        for k in list(self.data):
            if k.startswith(prefix):
                yield k


@pytest.fixture
def fake_redis(monkeypatch):
    fr = FakeRedis()
    monkeypatch.setattr(tc, "_redis", lambda: fr)
    monkeypatch.setattr(tc, "_last_fail", 0.0)  # 清除跨用例冷却状态
    return fr


def test_normalize_query_variants():
    assert tc.normalize_query("请问退货时限几天？") == "退货时限几天"
    assert tc.normalize_query("退货时限几天呢") == "退货时限几天"
    assert tc.normalize_query("  退货政策  ") == "退货政策"
    # 全角 → 半角
    assert tc.normalize_query("请问：退货时限？") == "退货时限"


def test_normalize_keeps_meaning_distinct():
    # 语义不同的问题不因清洗合并（零误命中）
    assert tc.normalize_query("退货政策") != tc.normalize_query("退款政策")


def test_normalize_keeps_action_prefix():
    """动作性前缀（帮我/麻烦）不剥离：防止"帮我退货"归一成"退货"，
    与新会话业务流消息（本应进 RETURN/REFUND 状态机）撞缓存 key 被短路成政策答复。"""
    assert tc.normalize_query("帮我查一下退货政策") == "帮我查一下退货政策"
    assert tc.normalize_query("帮我退货") == "帮我退货"
    # 动作消息与政策咨询不同 key，绝不相撞
    assert tc.normalize_query("帮我退货") != tc.normalize_query("退货")
    assert tc.normalize_query("帮我退货") != tc.normalize_query("退货政策")
    # 纯询问客套仍剥离（政策咨询变体命中不受影响）
    assert tc.normalize_query("请告诉我退货政策") == "退货政策"


def test_turn_key_stable_and_model_sensitive(monkeypatch):
    assert tc.turn_key(tc.normalize_query("退货政策")) == tc.turn_key(tc.normalize_query("请问退货政策"))
    # 换模型 → key 变化（旧答案失效）
    k0 = tc.turn_key(tc.normalize_query("退货政策"))
    monkeypatch.setattr(tc.settings, "deepseek_model_chat", "other-model")
    assert tc.turn_key(tc.normalize_query("退货政策")) != k0


@pytest.mark.asyncio
async def test_get_set_roundtrip(fake_redis):
    key = tc.turn_key(tc.normalize_query("退货政策"))
    payload = {"v": 2, "intent": "POLICY_INQUIRY", "reply": "7 天内",
               "search_policy": {"ok": True, "data": {"results": []}, "error": None}}
    await tc.set(key, payload, ttl=600)
    assert await tc.get(key) == payload


@pytest.mark.asyncio
async def test_get_version_mismatch_returns_none(fake_redis):
    await fake_redis.set("cs:turn:x", '{"v": 99, "reply": "未来格式"}')
    assert await tc.get("cs:turn:x") is None


@pytest.mark.asyncio
async def test_get_v1_legacy_cache_returns_none(fake_redis):
    """v=1 旧缓存（search_policy 裸 dict）随版本 bump 全部失效，防旧结构误读新契约。"""
    await fake_redis.set("cs:turn:legacy",
                         '{"v": 1, "intent": "POLICY_INQUIRY", "reply": "旧答案",'
                         ' "search_policy": {"results": []}}')
    assert await tc.get("cs:turn:legacy") is None


@pytest.mark.asyncio
async def test_get_missing_returns_none(fake_redis):
    assert await tc.get("cs:turn:nope") is None


@pytest.mark.asyncio
async def test_redis_error_silent_miss(monkeypatch):
    def boom_redis():
        raise ConnectionError("no redis")

    monkeypatch.setattr(tc, "_redis", boom_redis)
    monkeypatch.setattr(tc, "_last_fail", 0.0)
    assert await tc.get("cs:turn:x") is None  # 不抛异常


@pytest.mark.asyncio
async def test_cooldown_skips_redis_after_failure(monkeypatch):
    calls = {"n": 0}

    def boom_redis():
        calls["n"] += 1
        raise ConnectionError("no redis")

    monkeypatch.setattr(tc, "_redis", boom_redis)
    monkeypatch.setattr(tc, "_last_fail", 0.0)
    assert await tc.get("k") is None
    assert calls["n"] == 1  # 首次失败尝试
    assert tc._disabled() is True  # 冷却生效
    assert await tc.get("k") is None
    assert calls["n"] == 1  # 冷却期不再尝试


@pytest.mark.asyncio
async def test_flush_all_only_turn_keys(fake_redis):
    await fake_redis.set("cs:turn:a", "1")
    await fake_redis.set("cs:turn:b", "1")
    await fake_redis.set("rag_cache:x", "1")  # 不相关前缀不清
    await tc.flush_all()
    assert "cs:turn:a" not in fake_redis.data
    assert "cs:turn:b" not in fake_redis.data
    assert "rag_cache:x" in fake_redis.data
