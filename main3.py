from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", api_key)

# Initialize OpenAI model
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.5)

# Ask the model something
result = model.invoke("What's up?")
print("Answer:", result.content)  # .content gives just the text

for token in model.stream:
    time.sleep(2)
    print("what is life", token)
