import sqlite3

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from nodes import get_schema
from scenario import Scenario

if __name__ == "__main__":
    db_path = "example.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # Here is our checkpointer
    memory = SqliteSaver(conn)

    # getting design-time schema
    schema = get_schema()

    # compiling scenario object
    sc = Scenario(schema, memory)
    sc.compile()

    config = {"configurable": {"thread_id": "1"}}
    # ЕСЛИ ПЕРЕЗАПУСТИТЬ И ПЕРЕДАТЬ ЭТОТ ЖЕ thread_id - КОНТЕКСТ СЦЕНАРИЯ СОХРАНИТСЯ С МОМЕНТА

    input_message = HumanMessage(content="hi! I'm Lance")
    # OR
    # input_message = HumanMessage(content="how much characters in my name?")

    response = sc.run(input_message, config)
    response.pretty_print()
    # run 1 (Hi Lance! How can I assist you today?)
    # OR
    # run 2 (Your name "Lance" has 5 characters.)
