"""文本预清洗纯函数测试（移植 good-question test_cleaner + customer-service markdown 场景）。"""
from app.rag.text_cleaner import clean_text


def test_remove_control_chars():
    assert clean_text("a\x00b") == "ab"


def test_collapse_blank_lines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_strip_whitespace():
    assert clean_text("  hello  ") == "hello"


def test_fullwidth_space_to_half():
    assert clean_text("a　b") == "a b"


def test_trim_trailing_spaces():
    assert clean_text("hello   \nworld") == "hello\nworld"


def test_empty_text():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_normal_text_unchanged():
    assert clean_text("微服务架构设计指南") == "微服务架构设计指南"


def test_fullwidth_to_halfwidth():
    assert clean_text("ＡＢＣ１２３＠") == "ABC123@"


def test_chinese_punctuation_preserved():
    # 中文全角标点应保留（NFKC 不影响中文标点）
    assert clean_text("你好，世界。") == "你好，世界。"
    assert clean_text("退货政策？") == "退货政策？"


def test_unify_newlines():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_remove_bom():
    assert clean_text("﻿开头文本") == "开头文本"


def test_zero_width_space_removed():
    # 零宽空格不可见但会打断 embedding 中文切分（"工资​发放"），必须删除
    assert clean_text("工资​发放") == "工资发放"


def test_zero_width_joiners_removed():
    assert clean_text("a‍b‌c") == "abc"


def test_nbsp_to_space():
    assert clean_text("a b") == "a b"


def test_inline_bom_removed():
    # 行中 BOM（非行首）也要删除，不能只 lstrip
    assert clean_text("a﻿b") == "ab"


def test_markdown_heading_preserved():
    # splitter 依赖 # 标题切 section，清洗必须保留
    assert clean_text("# 退货政策\n\n## 退货时限\n\n正文") == "# 退货政策\n\n## 退货时限\n\n正文"


def test_markdown_table_preserved():
    # 表格 | 分隔结构保留（清洗不动 |，只做宽度/空白规范化）
    md = "| 项目 | 说明 |\n| --- | --- |\n| 退货 | 7 天内 |"
    assert clean_text(md) == md


def test_markdown_code_fence_preserved():
    # ``` 代码块 fence 保留；全角字符转半角（代码本就应半角，属正确行为）
    cleaned = clean_text("```python\nx = 'Ａ'\n```")
    assert cleaned.startswith("```python\n")
    assert cleaned.endswith("\n```")
    assert "Ａ" not in cleaned


def test_repeated_lines_disabled_by_default():
    """页眉/页脚去重默认关闭：规则未经验证，误删风险不能进默认路径"""
    header = "机密资料"
    text = "\n".join([header] * 7)
    assert clean_text(text) == text


def test_repeated_short_lines_removed_when_enabled():
    """开启后：短行（≤30 字符）重复 ≥5 次视为页眉/页脚噪声只留首行；长行正文全保留"""
    header = "机密资料"  # 5 字符短行，重复 7 次
    body = "这是一段超过三十个字符长度的正常正文长文本行，用于验证不受去重影响。"  # 长行
    lines = [header, body] * 3 + [header, body, header]
    text = "\n".join(lines)
    cleaned = clean_text(text, dedupe_repeated_lines=True)
    assert cleaned.count(header) == 1, "短行重复 ≥5 次应只保留首行"
    assert cleaned.count(body) == 4, "长行（正文）不应被去重"


def test_short_repeated_below_threshold_kept():
    """重复次数 < 5 的短行保留（阈值保守，避免误删正文中少量重复的短语）"""
    header = "注意"  # 只重复 3 次 < 5
    text = "\n".join(["正文", header] * 3)
    cleaned = clean_text(text, dedupe_repeated_lines=True)
    assert cleaned.count(header) == 3
