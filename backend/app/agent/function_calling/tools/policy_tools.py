"""政策检索工具。内部调用 RAG Retriever。"""
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
                "source": r.metadata.get("source", ""),
            }
            for r in results
        ]
    }
