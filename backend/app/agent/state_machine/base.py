"""LangGraph 状态机基础。

设计：对话型状态机按"每轮用户输入推进一个节点"执行。
- 图结构: START → route(读 state.stage) → 目标节点 → END
- 节点函数返回 state 的部分更新（LangGraph 自动合并），并推进 state.stage
- 需要用户输入的节点若输入不足，保持 stage 不变并返回追问 message，
  下一轮输入重新进入同一节点
"""
from typing import Callable, Type

from langgraph.graph import END, START, StateGraph


class BaseStateMachine:
    # 子类覆盖: 节点名 → 节点函数
    NODES: dict[str, Callable] = {}
    # 子类覆盖: TypedDict 状态类型
    STATE_TYPE: Type = dict

    def __init__(self) -> None:
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(self.STATE_TYPE)
        for name, fn in self.NODES.items():
            builder.add_node(name, fn)
        # 路由: 根据 state.stage 决定当前执行哪个节点
        builder.add_conditional_edges(START, self._route, self._mapping())
        for name in self.NODES:
            builder.add_edge(name, END)
        return builder.compile()

    def _route(self, state: dict) -> str:
        return state.get("stage") or next(iter(self.NODES))

    def _mapping(self) -> dict:
        mapping = {name: name for name in self.NODES}
        mapping[END] = END
        return mapping

    async def step(self, state: dict, user_input: str) -> dict:
        """单轮推进：注入用户输入，循环执行非输入节点直到停在"等待输入"节点或终态。

        节点返回 awaiting 标记表示等待用户输入（其 message 即追问/确认文案）。
        """
        result = {**state, "user_input": user_input}
        for _ in range(20):  # 保险上限，防节点异常导致死循环
            result = await self.graph.ainvoke(result)
            if result.get("final") or result.get("awaiting"):
                return result
        return {**result, "final": True, "message": "流程异常，请重新发起请求"}
