"""切分器单测（mock tokenizer 为字符近似，不依赖模型文件/网络）。"""
import pytest

from app.rag import splitter
from app.rag.splitter import chunk_document


class _FakeTokenizer:
    """字符近似编码器：1 中文字符 ≈ 1 token，避免测试加载真实 BGE 模型。"""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


@pytest.fixture(autouse=True)
def _fake_tokenizer(monkeypatch):
    monkeypatch.setattr(splitter, "_get_tokenizer", lambda: _FakeTokenizer())


def test_split_sections_heading_stack():
    md = "# 退换货政策\n\n基础规则。\n\n## 退货时限\n\n7 天内可退。\n\n# 退款政策\n\n按原路退回。"
    sections = splitter._split_markdown_sections(md)
    assert [s["heading_path"] for s in sections] == [
        ["退换货政策"], ["退换货政策", "退货时限"], ["退款政策"],
    ]
    assert [s["heading_level"] for s in sections] == [1, 2, 1]


def test_chunks_carry_heading_path():
    md = "# 退换货政策\n\n基础规则。\n\n## 退货时限\n\n签收后 7 天内。\n\n## 运费规则\n\n非质量问题运费自理。"
    chunks = chunk_document(md, "return_policy")
    paths = {tuple(c["metadata"]["heading_path"]) for c in chunks}
    assert ("退换货政策",) in paths
    assert ("退换货政策", "退货时限") in paths
    assert ("退换货政策", "运费规则") in paths
    # 内容归属正确：heading_path 含「退货时限」的 chunk 是该项正文
    for c in chunks:
        if c["metadata"]["heading_path"] == ["退换货政策", "退货时限"]:
            assert "7 天内" in c["content"]


def test_plain_text_single_section():
    chunks = chunk_document("第一段。\n\n第二段。", "doc")
    assert chunks
    assert all(c["metadata"]["heading_path"] == [] for c in chunks)
    assert all(c["metadata"]["heading_level"] == 0 for c in chunks)
    assert all(c["metadata"]["splitter"] == "sentence_splitter" for c in chunks)


def test_metadata_fields_present():
    chunks = chunk_document("# 标题\n\n正文内容。", "src")
    m = chunks[0]["metadata"]
    # section_id = source:<heading_path sha8>:occurrence（内容级稳定，非顺序索引）
    assert m["section_id"].startswith("src:") and m["section_id"].endswith(":0")
    assert m["heading_path"] == ["标题"]
    assert m["heading_level"] == 1
    assert m["chunk_index"] == 0
    assert m["total_chunks"] == len(chunks)
    assert m["token_count"] > 0
    assert m["source_type"] == "paragraph"


def test_long_paragraph_split_by_sentence():
    # 单段超长文本（无段落分隔）→ 按中文句末标点二次切，不产生超 chunk_size 的块
    sent = "本商品自签收之日起 7 天内支持无理由退货，运费由买家承担。"
    chunks = chunk_document(sent * 20, "doc")
    assert len(chunks) > 1
    for c in chunks:
        # 假 tokenizer 下 token 数 = 字符数
        assert len(c["content"]) <= splitter.CHUNK_SIZE + splitter.OVERLAP


def test_paragraph_boundary_preferred():
    para_a = "甲" * 400 + "。"
    para_b = "乙" * 400 + "。"
    chunks = chunk_document(para_a + "\n\n" + para_b, "doc")
    assert len(chunks) >= 2  # 段落边界优先于字符硬切
    joined = "".join(c["content"] for c in chunks)
    assert "甲" * 400 in joined and "乙" * 400 in joined  # 内容完整无丢失


def test_empty_content_returns_empty():
    assert chunk_document("  \n ", "doc") == []


def test_max_chunks_truncation(monkeypatch):
    monkeypatch.setattr(splitter, "CHUNK_SIZE", 4)
    monkeypatch.setattr(splitter, "OVERLAP", 1)  # SentenceSplitter 要求 overlap < chunk_size
    monkeypatch.setattr(splitter, "MAX_CHUNKS", 3)
    chunks = chunk_document("内容。" * 100, "doc")
    assert 0 < len(chunks) <= 3


def test_code_fence_not_split_as_heading():
    md = "# 说明\n\n```python\n# 这是代码注释，不是标题\nx = 1\n```\n\n## 结尾\n\n正文。"
    chunks = chunk_document(md, "doc")
    paths = {tuple(c["metadata"]["heading_path"]) for c in chunks}
    # 代码块内 # 不产生单独 section
    assert ("说明", "这是代码注释，不是标题") not in paths
    assert ("说明", "结尾") in paths
    # 代码块内容完整保留（fence 行进 content，未被剔除）
    joined = "".join(c["content"] for c in chunks)
    assert "x = 1" in joined


def test_section_id_stable_across_edits():
    # 文档开头新增 section，不影响既有章节的 section_id（heading_path 内容级稳定）
    md1 = "# 退换货政策\n\n规则一。\n\n## 退货时限\n\n7 天内。\n"
    md2 = "# 开头新增\n\n新内容。\n\n# 退换货政策\n\n规则一。\n\n## 退货时限\n\n7 天内。\n"

    def sid_of(chunks, path):
        for c in chunks:
            if c["metadata"]["heading_path"] == path:
                return c["metadata"]["section_id"]
        return None

    assert sid_of(chunk_document(md1, "doc"), ["退换货政策", "退货时限"]) == \
        sid_of(chunk_document(md2, "doc"), ["退换货政策", "退货时限"])


def test_chunk_content_cleaned_fullwidth_and_zero_width():
    """复制粘贴脏文本：分块前清洗全角数字（７→7）并删除零宽字符（会打断 BGE 中文切分）。"""
    dirty = "## 退货时限\n\n自签收之日起 ７ 天内可退。工资​发放规则见政策。"
    chunks = chunk_document(dirty, "doc")
    texts = [c["content"] for c in chunks]
    assert any("7 天" in t for t in texts)
    assert not any("７" in t or "​" in t for t in texts)
    # 标题结构不受清洗影响
    assert ("退货时限",) in {tuple(c["metadata"]["heading_path"]) for c in chunks}
