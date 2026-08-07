"""文本分块：滑窗 + overlap。

按字符近似 512 token + 128 overlap（中文一个字符 ≈ 1 token）。
对政策 Markdown 文档够用；后续可换 tokenizer 精确分块。
"""

CHUNK_SIZE = 512
OVERLAP = 128


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks
