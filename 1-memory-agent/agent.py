from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from tools import TOOLS

load_dotenv(find_dotenv())

# LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
# LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2")
# LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT")
# OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
# GIGACHAT_API_KEY: str = os.getenv("GIGACHAT_API_KEY")
MODEL: str = "gpt-4o-mini"

llm = ChatOpenAI(model=MODEL)
llm_with_tools = llm.bind_tools(TOOLS)

# System message
SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)

# Node


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([SYSTEM_MESSAGE] + state["messages"])]}


def init_graph() -> StateGraph:
    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    return builder


if __name__ == "__main__":
    graph = init_graph()

    memory = MemorySaver()
    react_graph_memory = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "1"}}

    # Specify an input
    msg1 = [HumanMessage(content="Add 3 and 4.")]

    msg1 = react_graph_memory.invoke({"messages": msg1}, config)
    for m in msg1["messages"]:
        m.pretty_print()

    msg2 = [HumanMessage(content="Multiply that by 8.")]
    msg2 = react_graph_memory.invoke({"messages": msg2}, config)
    for m in msg2["messages"]:
        m.pretty_print()

    msg3 = [HumanMessage(content="Divide by 6.")]
    msg3 = react_graph_memory.invoke({"messages": msg3}, config)
    for m in msg3["messages"]:
        m.pretty_print()
