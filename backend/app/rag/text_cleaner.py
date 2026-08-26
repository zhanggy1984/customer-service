"""文本预清洗与内容规范化（参考 good-question utils/text_cleaner）。

在分块/向量化前对文档内容做确定性规范化，提升 embedding 与检索质量：
- 去 BOM、统一换行符
- 全角字母/数字/符号转半角（NFKC），但跳过中文标点保留中文排版
- 移除控制字符与不可见噪声（零宽空格会打断 BGE 中文切分："工资​发放"被拆成两段）
- 压缩空白与空行、行尾空白
- （可选，默认关）页眉/页脚/页码重复短行去重——对上传文档预留，规则未经验证不进默认路径

设计边界：
- 只动字符宽度/空白/控制字符，不删正文、不改标点语义；
- 不破坏 Markdown 结构：标题 `#`、表格 `|`、代码块 ```` ``` ```` 原样保留（splitter 依赖这些结构）；
- 清洗作用于派生向量侧（chunk 生成前），MySQL knowledge_docs 保留原始 markdown（source of truth）。
"""
import re
import unicodedata

# 页眉/页脚/页码去重的保守阈值（同 good-question）：
# - 只处理 ≤ _REPEAT_MAX_LEN 字符的短行（页眉/页脚/页码都是短行，长行是正文）
# - 全文出现 ≥ _REPEAT_MIN_COUNT 次才视为跨页重复噪声（阈值取大，宁可漏删不可误删）
_REPEAT_MAX_LEN = 30
_REPEAT_MIN_COUNT = 5

# 中文全角标点：NFKC 会把这些一并转半角、破坏中文排版，故跳过不转
_CHINESE_PUNCT = set("，。！？；：、（）《》「」『』【】“”‘’—…")


def _remove_repeated_lines(text: str) -> str:
    """移除跨页重复的短行（页眉/页脚/页码噪声），保留首次出现。

    逐页抽取的 PDF 会把每页页眉（公司名/章节名）、页脚（页码/版权）重复 N 次；
    不清理会产出语义冗余 chunk，污染向量库与 rerank 候选。
    启发式：短行且全文出现 ≥ _REPEAT_MIN_COUNT 次视为噪声，只保留首次出现；
    正文中合法重复的短短语同样保留一份，不丢信息；长行（正文）不受影响。
    """
    from collections import Counter

    lines = text.split("\n")
    counts = Counter(lines)
    seen: set[str] = set()
    keep: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped
            and len(stripped) <= _REPEAT_MAX_LEN
            and counts[line] >= _REPEAT_MIN_COUNT
            and line in seen
        ):
            continue  # 噪声重复行：首次出现已保留，跳过
        keep.append(line)
        seen.add(line)
    return "\n".join(keep)


def clean_text(text: str, dedupe_repeated_lines: bool = False) -> str:
    """预清洗：规范化文本，保持段落结构（\n）。

    dedupe_repeated_lines：页眉/页脚/页码重复短行去除（保守启发式，默认关闭）。
    默认关：规则依赖阈值猜测，误删风险需真实样本/评测基线验证后才默认开启
    （good-question solution.md 13.3 的教训：拍脑袋优化会倒退）。
    """
    if not text:
        return ""

    # 1. 去 BOM（行首）
    text = text.lstrip("﻿")

    # 2. 统一换行符（\r\n / \r → \n）
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. 全角字符转半角（NFKC：ＡＢＣ→ABC、１２３→123、＠→@）
    #    注意：NFKC 会把中文标点（，。！？等全角形式）一并转半角，破坏中文排版，
    #    故跳过中文标点集合，只对字母/数字/符号规范化。
    text = "".join(
        unicodedata.normalize("NFKC", ch) if ch not in _CHINESE_PUNCT else ch
        for ch in text
    )

    # 4. 移除控制字符（保留换行和制表符）
    text = "".join(ch for ch in text if ch >= " " or ch in "\n\t")

    # 5. 不可见噪声：零宽字符（U+200B/C/D）删除——不可见但会打断 embedding tokenizer
    #    的中文切分（"工资​发放"被拆成两段，检索精度受损）；NBSP(U+00A0) 转普通
    #    空格；行中 BOM(U+FEFF) 删除（行首的已在第 1 步 lstrip）。
    text = text.replace("​", "").replace("‌", "").replace("‍", "")
    text = text.replace("﻿", "")
    text = text.replace(" ", " ")

    # 6. 全角空格转半角（NFKC 已处理，这里兜底）
    text = text.replace("　", " ")

    # 7. 压缩连续空行为最多两个换行（保留段落分隔）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 8. 行尾多余空白
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)

    # 9. 页眉/页脚/页码重复短行去除（保守启发式，默认关闭，见函数 docstring）
    if dedupe_repeated_lines:
        text = _remove_repeated_lines(text)

    return text.strip()
