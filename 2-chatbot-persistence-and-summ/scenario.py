from typing import Any

from IPython.display import Image, display
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, StateSnapshot


class Scenario:
    compiled: CompiledStateGraph

    schema: StateGraph
    memory: Checkpointer

    def __init__(self, schema: StateGraph, memory: Checkpointer):
        self.schema = schema
        self.memory = memory

    def compile(self):
        self.compiled = self.schema.compile(checkpointer=self.memory)

    def show_graph(self):
        if self.compiled is not None:
            display(Image(self.graph.get_graph().draw_mermaid_png()))

    def run(self, message: HumanMessage, config: dict[str, Any]) -> BaseMessage:
        output = self.compiled.invoke({"messages": [message]}, config)

        if len(output["messages"]) == 0:
            return AIMessage("no result messages")

        return output["messages"][-1]

    def get_state(self, config: dict[str, Any]) -> StateSnapshot:
        return self.compiled.get_state(config)
