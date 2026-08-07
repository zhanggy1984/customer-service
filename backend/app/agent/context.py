"""上下文装配：把 RAG 检索结果装配成 LLM 可用的政策上下文。"""
from app.rag.retriever import retriever


async def search_policy_context(query: str) -> tuple[str, list]:
    """检索政策知识库，返回 (上下文文本, 检索结果列表)。空结果返回空串。"""
    results = await retriever.search(query)
    if not results:
        return "", []
    ctx = "\n\n".join(f"[{r.metadata.get('source', '')}] {r.text}" for r in results)
    return ctx, results
