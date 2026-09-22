"""提示词模板集中存放与加载。

每个提示词一个纯文本文件（.md），正文即全部内容、零转义——改文案不必动代码。
占位符约定：
- 无占位符：直接读原文使用；
- 占位符用 `{name}`，由调用方 `str.format()` 填充（如 decision.md 的 `{tools}`）；
- 模板正文含**字面大括号**（JSON 示例）时，改用 `string.Template` + `$name`，
  否则 `format` 会把字面大括号当占位符——`intent_system.md` 即属此类。

模板文件放在包目录内，随 `COPY backend/ /app/` 进镜像。
"""
from pathlib import Path

_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """读取提示词模板正文（不做任何插值）。

    Args:
        name: 模板名（不含 .md 后缀），如 "policy_answer"。

    Returns:
        模板正文；占位符原样保留，由调用方填充。
    """
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")
