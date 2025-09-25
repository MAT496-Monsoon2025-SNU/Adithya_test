from dotenv import load_dotenv
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv(override=True)

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

# Original Python functions
def XOR_func(a: int, b: int) -> int:
    return a ^ b

def decimal_to_binary(decimal: int) -> str:
    arr = []
    if decimal == 0:
        return "0"
    while decimal > 0:
        arr.append(str(decimal % 2))
        decimal //= 2
    return ''.join(reversed(arr))

# LangChain tools
@tool
def XOR(a: int, b: int) -> int:
    """XOR two integers and return the result."""
    return XOR_func(a, b)

@tool
def decimal_to_binary_tool(decimal: int) -> str:
    """Convert a decimal number to its binary representation."""
    return decimal_to_binary(decimal)

# Map tool names to original Python functions
original_funcs = {
    "XOR": XOR_func,
    "decimal_to_binary_tool": decimal_to_binary
}

tools_list = [XOR, decimal_to_binary_tool]
llm_with_tools = llm.bind_tools(tools_list)

question = "turn 4 and 3 to binary. What is 3 xor 4?"
response = llm_with_tools.invoke(question)

print("\n--- Raw LLM Response ---")
print(response)

if response.tool_calls:
    for tool_call in response.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        # Automatically call the correct original function
        if name in original_funcs:
            result = original_funcs[name](**args)
            print(f"\n--- Tool Result ({name}) ---")
            print(result)
else:
    print("No tool calls found in the response.")
