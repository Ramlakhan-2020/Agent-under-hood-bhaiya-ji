import os
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage

Max_Iteration=3
MODEL="qwen3:4b"

# creating a tools name

@tool
def get_product_price(product:str)->float:
    """Look up the price of a product by its name"""
    print(f"product price = {product}")
    prices = {"laptop":10000, "headphones":2000, "keyboard":500}
    return prices.get(product,0)

@tool
def apply_discount(price:float,discount:str)->float:
    """Apply a discount to a price and return final price"""
    print(f"price = {price}, discount = {discount}")
    discount_percentage={"bronze":5, "silver":10, "gold":15}
    discount_value=discount_percentage.get(discount,0)
    return round(price*(1-discount_value/100),2)

@traceable(name="langchain agent loop")
def run_agent(question:str):
    tools=[get_product_price, apply_discount]
    tools_dict={t.name:t for t in tools}
    llm=init_chat_model(f'ollama:{MODEL}', temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages=[
        SystemMessage(
            content=(
                "Strict rules apply:"
                "Never guess and assume any product price"
                "Never guess the product name"
                "You must call get_product_price to find the price"
                "Only call apply_discount after you have received a price"
                "do not use your own discount math use apply_discount tools always"
                "if user does not pass discount"
                "asked them which discount to use-- do not assume one"
            )
        ),
        HumanMessage(content=question)
    ]
    for iteration in range(1,Max_Iteration+1):
        print(f"iteration = {iteration}")
        ai_message= llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        print(f"tool_calls = {tool_calls}")
        if not tool_calls:
            print(f"\n Final answer = {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_id = tool_call.get("id")
        tool_to_use = tools_dict.get(tool_name)
        print(f"tool_to_use = {tool_to_use}")

        if tool_to_use is None:
            raise ValueError(f"tools {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"observation = {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=f"Tool result for {tool_name}: {observation}",tool_call_id=tool_id)
        )

    print("error: max iteration reached")
    return None


if __name__ == "__main__":
    print("Hello langchain agent")
    result=run_agent("What is the price of a laptop after apply gold discount?")
    print("\n result = ", result)