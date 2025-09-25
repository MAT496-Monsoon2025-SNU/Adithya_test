from dotenv import load_dotenv
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv(override=True)

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

# Original Python function
def multiply_func(a: float, b: float) -> float:
    return a * b

def divide_func(a: float, b: float) -> float:
    return a / b

# Decorated tool for LangChain
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return multiply_func(a, b)

@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers and return the result."""
    return divide_func(a, b)

tools_list = [multiply, divide]
tools_dict = {t.name: t for t in tools_list}

llm_with_tools = llm.bind_tools(tools_list)

question = "What is 10 times 40?"
question += " Also, what is 400 divided by 20?"
question += "what is 10 plus 40"
response = llm_with_tools.invoke(question)

print("\n--- Raw LLM Response ---")
print(response)

if response.tool_calls:
  for tool_call in response.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        # Call the original Python function to get the numeric result
        if name == "multiply":
            result = multiply_func(**args)
        elif name == "divide":
            result = divide_func(**args)
        else:
            result = None
        print(f"\n--- Tool Result ({name}) ---")
        print(result)
else:
    print("No tool calls found in the response.")
