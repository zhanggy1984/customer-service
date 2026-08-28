"""政策检索工具。内部调用 RAG Retriever。

统一返回信封 {ok, data, error}（FC 契约）：data 含 results/source_count/max_score 聚合字段，
供决策 LLM 判断"是否已检索到足够依据"与观测层聚合（对应 good-question 的 source_count 思路）。
"""
from app.infrastructure import (
    RetrievalUnavailableError,
    normalize_query,
    retriever,
    source_label,
)

_QUERY_MAX_LEN = 100  # LLM 可能传整段对话，超长前缀截断（简单截断，不引入切句逻辑）


async def search_policy(params: dict, user_id: int, session_id: str) -> dict:
    raw = (params.get("query") or "").strip()
    if not raw:
        return {"ok": False, "data": None, "error": {"code": "missing_query", "message": "缺少 query 参数"}}
    # 入口清洗：复用 turn_cache.normalize_query（刻意不剥"帮我"动作前缀，防撞状态机 key）。
    # 空回退原文：纯客套 query 归一后为空，回退保证检索 query 非空（防空 query 拖垮召回）。
    q = normalize_query(raw) or raw
    if len(q) > _QUERY_MAX_LEN:
        q = q[:_QUERY_MAX_LEN]
    try:
        results = await retriever.search(q)
    except RetrievalUnavailableError:
        # 检索故障（Milvus/embedding 不可用）≠ 检索空：映射专用错误码，
        # 消费端据此走 LLM 兜底，不误报"未收录知识库"。
        return {"ok": False, "data": None,
                "error": {"code": "retrieval_unavailable", "message": "知识库检索暂不可用"}}
    scores = [r.score for r in results]
    return {
        "ok": True,
        "data": {
            "results": [
                {
                    "text": r.text,
                    "score": round(r.score, 3),
                    "source": source_label(r.metadata),  # 带标题路径溯源，SSE 透出可观测
                }
                for r in results
            ],
            "source_count": len(results),
            "max_score": round(max(scores), 3) if scores else None,
        },
        "error": None,
    }
