"""政策检索工具。内部调用 RAG Retriever。"""
from app.rag.interfaces import source_label
from app.rag.retriever import retriever


async def search_policy(params: dict, user_id: int, session_id: str) -> dict:
    query = params.get("query")
    if not query:
        return {"error": "缺少 query 参数"}
    results = await retriever.search(query)
    return {
        "results": [
            {
                "text": r.text,
                "score": round(r.score, 3),
                "source": source_label(r.metadata),  # 带标题路径溯源，SSE 透出可观测
            }
            for r in results
        ]
    }
