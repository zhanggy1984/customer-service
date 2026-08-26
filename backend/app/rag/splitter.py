"""文本分块：标题层级切 section + LlamaIndex SentenceSplitter 段落切分（参考 good-question chunker）。

改造前为字符滑窗（512+128 近似 token），完全忽略 Markdown 标题结构——政策文档的
「## 退货时限」语义常被硬切截断。现改为：
- Markdown 标题（# 层级）先切成 section（语义单元），chunk 带 heading_path 溯源路径；
  代码块 fence（```）内不按标题切，避免误伤代码注释里的 #；
- section 内用 SentenceSplitter 段落切分：主分隔符 \\n\\n（段落），超长段内按中文句末标点二次切；
- BGE tokenizer 精确计数。chunk_size=448 留 12.5% 余量：bge-small-zh-v1.5 max_seq_length=512，
  恰在边界加 [CLS]/[SEP] 后会被 truncation 掐尾，否定词（如"不予退货"）落在边界会语义翻转。
"""
import hashlib
import os
import re
from functools import lru_cache

from app.config import settings
from app.rag import text_cleaner
from app.utils.logger import logger

# 448 = bge-small-zh-v1.5 max_seq_length(512) 留 12.5% 余量，见模块 docstring
CHUNK_SIZE = 448
OVERLAP = 128
# 单文档 chunk 数上限：防极端大文件拖垮向量化与存储
MAX_CHUNKS = 2000

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@lru_cache(maxsize=1)
def _get_tokenizer():
    """BGE tokenizer（精确 token 计数），模块级缓存。复用 HF 缓存离线加载（与 embedder 同目录）。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoTokenizer

    # embedder 已把模型仓库（含 tokenizer）拉到 HF_HUB_CACHE，显式指同目录离线读即可
    return AutoTokenizer.from_pretrained(settings.embedding_model, cache_dir=HF_HUB_CACHE)


def _count_tokens(text: str) -> int:
    """按 BGE tokenizer 统计 token 数。"""
    return len(_get_tokenizer().encode(text))


def _split_markdown_sections(text: str) -> list[dict]:
    """按 Markdown 标题层级切 section，返回 [{heading_path, heading_level, content}]。

    heading_stack 全局延续：子节归入祖先标题路径，如「退货时限」→ ["退换货政策", "退货时限"]。
    无标题文本天然单 section（heading_path=[]、heading_level=0）。
    ``` 代码块 fence 内不按标题切（代码注释里的 # 会误伤），fence 行本身进 content 保持原文。
    """
    sections: list[dict] = []
    heading_stack: list[tuple[int, str]] = []
    current: dict | None = None
    in_code = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code  # 代码块开关（fence 行本身也进 content 保持原文）
            if current is None:
                current = {"heading_path": [], "heading_level": 0, "content": ""}
            current["content"] += line + "\n"
            continue
        if in_code:
            if current is None:
                current = {"heading_path": [], "heading_level": 0, "content": ""}
            current["content"] += line + "\n"
            continue
        m = _HEADING_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # 维护标题栈：同层或更高层标题出栈（子标题入栈）
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            if current is not None:
                sections.append(current)
            current = {
                "heading_path": [t for _, t in heading_stack],
                "heading_level": level,
                "content": "",
            }
        else:
            if current is None:
                current = {"heading_path": [], "heading_level": 0, "content": ""}
            current["content"] += line + "\n"

    if current is not None:
        sections.append(current)
    return sections


def _section_id(source: str, heading_path: list[str], occurrence: int) -> str:
    """内容级稳定 section_id：基于 heading_path 的 hash，文档其他位置增删不漂移。

    顺序索引（source:section_idx）在文档任意增删标题/段落后会整体漂移，导致新旧数据
    的 section_id 对不上、章节扩充静默退化。heading_path 是内容级的：只要该章节标题
    文本不变，其 section_id 就稳定。同 heading_path 重复出现时用 occurrence 区分。
    无标题文本统一归 source:plain:{n}（通常整篇单 section）。
    """
    if not heading_path:
        return f"{source}:plain:{occurrence}"
    digest = hashlib.sha256(">".join(heading_path).encode("utf-8")).hexdigest()[:8]
    return f"{source}:{digest}:{occurrence}"


def _infer_source_type(content: str) -> str:
    """粗判 chunk 内容类型（溯源展示用）。"""
    stripped = content.strip()
    if stripped.startswith(("|", "+")) and "|" in stripped:
        return "table"
    if any(l.lstrip().startswith(("- ", "* ", "1. ")) for l in stripped.splitlines()[:5]):
        return "list"
    if stripped.startswith(("def ", "class ", "import ", "```")):
        return "code"
    return "paragraph"


def chunk_document(content: str, source: str) -> list[dict]:
    """结构感知切分，返回 [{content, metadata}]。

    metadata 含 section_id/heading_path/heading_level/chunk_index/total_chunks/
    token_count/source_type/splitter。section_id 内容级稳定：有标题为
    "{source}:{sha8}:{occurrence}"（heading_path hash），无标题为 "{source}:plain:{occurrence}"，
    文档增删不漂移；供检索层按 (source, section_id) 回查同章节兄弟 chunk 扩充上下文。
    """
    # 预清洗（全角→半角/去零宽/统一换行等，见 text_cleaner）：脏文本（复制粘贴的
    # BOM/全角/零宽字符）会打断 BGE 中文切分、污染向量。清洗只作用于派生向量侧，
    # MySQL knowledge_docs 保留原始 markdown。splitter 依赖的标题/表格/代码块结构不受影响。
    content = text_cleaner.clean_text(content)
    content = content.strip()
    if not content:
        return []

    # 换 LlamaIndex SentenceSplitter（替换原字符滑窗）：主分隔符取段落（\\n\\n），
    # 超长段内再按中文句末标点切（secondary_chunking_regex）。
    # 坑：SentenceSplitter 内部用 re.findall 提取匹配块（非 split 切分），正则必须是
    # 「非分隔符块 + 可选分隔符」形态（默认 [^,.;]+[,.;]? 即此语义）；若只写 [。！？；\\n]
    # 会 findall 出纯标点、chunk 内容丢失。tokenizer 契约是 Callable[[str], list]
    # （_token_size 对返回值做 len()），返回 token ids 而非 int。
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        separator="\n\n",
        secondary_chunking_regex=r"[^。！？；\n]+[。！？；]?",
        tokenizer=lambda t: _get_tokenizer().encode(t),
    )

    results: list[dict] = []
    seen_path: dict[tuple, int] = {}  # heading_path -> 已出现次数：同 path 多 section 用 occurrence 区分
    for section in _split_markdown_sections(content):
        key = tuple(section["heading_path"])
        occurrence = seen_path.get(key, 0)
        seen_path[key] = occurrence + 1
        section_id = _section_id(source, section["heading_path"], occurrence)
        for chunk in splitter.split_text(section["content"]):
            chunk = chunk.strip()
            if not chunk:
                continue
            results.append({
                "content": chunk,
                "metadata": {
                    "section_id": section_id,
                    "heading_path": section["heading_path"],
                    "heading_level": section["heading_level"],
                    "source_type": _infer_source_type(chunk),
                    "token_count": _count_tokens(chunk),
                    "splitter": "heading_aware" if section["heading_path"] else "sentence_splitter",
                },
            })

    if len(results) > MAX_CHUNKS:
        logger.warning(
            "event=rag_chunk_truncated source=%s chunks=%s max=%s",
            source, len(results), MAX_CHUNKS,
        )
        results = results[:MAX_CHUNKS]

    total = len(results)
    for i, r in enumerate(results):
        r["metadata"]["chunk_index"] = i
        r["metadata"]["total_chunks"] = total
    return results
