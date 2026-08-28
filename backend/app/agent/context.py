"""上下文装配：把 RAG 检索结果装配成 LLM 可用的政策上下文。"""
from app.infrastructure import retriever, source_label


async def search_policy_context(query: str) -> tuple[str, list]:
    """检索政策知识库，返回 (上下文文本, 检索结果列表)。空结果返回空串。

    溯源前缀带标题路径（如 [return_policy > 退货时限]），LLM 可据此判断政策出处。
    """
    results = await retriever.search(query)
    if not results:
        return "", []
    ctx = "\n\n".join(f"[{source_label(r.metadata)}] {r.text}" for r in results)
    return ctx, results
