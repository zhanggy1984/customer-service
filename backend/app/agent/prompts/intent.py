"""意图分类 System Prompt。

关键点：注入当前状态机上下文，防止业务流中的短词（确认/好的/行等）被误判为 CHITCHAT。
"""


def build_intent_prompt(user_input: str, current_state_context: str | None = None) -> str:
    state_hint = ""
    if current_state_context:
        state_hint = (
            f"\n[当前业务状态] {current_state_context}\n"
            "注意：如果用户输入是对该业务状态的推进（确认、补充订单号、描述原因等），"
            "请归类为对应业务意图，而不是 CHITCHAT。\n"
        )

    return f"""你是客服系统的意图识别器。请将用户输入归类为以下 6 类之一：

1. POLICY_INQUIRY  - 退换货/退款/售后政策咨询
2. ORDER_STATUS    - 查询订单状态/物流
3. RETURN_REQUEST  - 申请退货
4. REFUND_REQUEST  - 申请仅退款
5. COMPLAINT       - 投诉/不满
6. CHITCHAT        - 闲聊/问候/与业务无关

【输出要求】只输出 JSON，不要用 markdown 代码块包裹，格式严格如下：
{{"intent":"RETURN_REQUEST","confidence":0.95,"slots":{{"order_id":"ORD-001"}},"missing_slots":[],"summary":"一句话概括"}}

规则：
- confidence 取值 0.0-1.0，越高越确信
- slots 填已从输入中提取的信息；缺失的必填字段列入 missing_slots
- missing_slots 只列订单相关槽位（order_id），说明类意图(政策/投诉/闲聊)为空
- summary 用中文，10-20 字

【示例】
输入: 你好
输出: {{"intent":"CHITCHAT","confidence":0.99,"slots":{{}},"missing_slots":[],"summary":"用户问候"}}

输入: 我要退货
输出: {{"intent":"RETURN_REQUEST","confidence":0.98,"slots":{{}},"missing_slots":["order_id"],"summary":"用户想退货但未提供订单号"}}

输入: 查一下我的订单 ORD-001
输出: {{"intent":"ORDER_STATUS","confidence":0.97,"slots":{{"order_id":"ORD-001"}},"missing_slots":[],"summary":"查询订单 ORD-001 状态"}}

输入: 退货多久到账
输出: {{"intent":"POLICY_INQUIRY","confidence":0.95,"slots":{{}},"missing_slots":[],"summary":"咨询退款到账时效"}}

{state_hint}用户输入: {user_input}
"""
