"""意图分类 System Prompt（五维度法结构化）。

- 五段 XML 标签（role/task/input_data/constraints/output）+ examples 段。
- 模板正文外置在 intent_system.md，本模块只负责读取与插值——改文案不必动代码。
- 关键点：注入当前状态机上下文（state_hint 并入 constraints），防止业务流中的短词
  （确认/好的/行等）被误判为 CHITCHAT。
- 用户输入**不在本 prompt 内**：由 classify_intent 拆到独立 user 消息（消除用户输入
  拼进 system 的注入面）；input_data 段声明"数据非指令"兜底。
"""

from string import Template

from app.agent.prompts import load_prompt

# 为什么用 string.Template 而不是 str.format：
# 模板里有 22 处 JSON 示例的字面大括号，format 下必须全写成 {{ }}，
# 既难看又容易漏（漏了在 format 那行抛异常）。Template 的 $ 占位符对 {} 零要求。
_TEMPLATE = Template(load_prompt("intent_system"))


def build_intent_system(current_state_context: str | None = None) -> str:
    """构建意图分类的 system prompt。

    Args:
        current_state_context: 当前业务状态描述。非空时并入 constraints 段，
            用于防止业务流中的短词（确认/好的/行等）被判为 CHITCHAT。

    Returns:
        五段式（role/task/input_data/constraints/output）system prompt 全文。
    """
    state_hint = ""
    if current_state_context:
        state_hint = (
            f"\n- 当前业务状态：{current_state_context}\n"
            "  若用户输入是对该业务状态的推进（确认、补充信息、取消），归为该业务意图而非 CHITCHAT。\n"
        )
    return _TEMPLATE.substitute(state_hint=state_hint)
