from json import tool
from typing import List,Dict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool

load_dotenv()

MODEL="qwen3:4b"

System_Prompt = """
   You are intelligent ai agent.
   Rules:
   - Think step by step.
   - Use tools when required.
   - Do not give random output use tools when needed.
   - Always return valid structured output.
"""

@tool
def get_search(genre:str)-> List[Dict]:
    """search book by genre"""
    database = {
        "sci-fi": [
            {"title": "Dune", "price": 699},
            {"title": "Foundation", "price": 450},
            {"title": "neuromancer", "price": 399},
            {"title": "Project Hail Mary", "price": 599},
            {"title": "Snow Crash", "price": 320},
        ]
    }
    return database.get(genre.lower(),[])

@tool
def filter_Book(books: List[Dict], budget:int)->List[Dict]:
    """filter books under budget"""
    return [book for book in books if book["price"]>=budget]


def run_agent(question:str, max_iteration=3):
    messages = [
        SystemMessage(content=System_Prompt),
        HumanMessage(content=question)
    ]

    # // tools creation

    tools = [get_search, filter_Book]

    tool_map = {
        tool.name: tool for tool in tools
    }

    # // create a model

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_tools = llm.bind_tools(tools)
    for iteration in range(max_iteration):
        print("iteration", iteration)
        response = llm_tools.invoke(messages)
        if not response.tool_calls:
            print("Final Value", response.content)
            return response.content
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_name=tool_call["name"]
            tool_args= tool_call["args"]
            tool_result = tool_map[tool_name].invoke(tool_args)
            print("Tool_result", tool_result)
            messages.append(
                ToolMessage(
                    content= str(tool_result),
                    tool_call_id = tool_call["id"],
                )
            )
    return "run-agent"

if __name__ == "__main__":
    print("Hello langchain agent")
    result = run_agent("Book recommendation above ₹500 based on sci-fi genre.")
    print(f"result", result)