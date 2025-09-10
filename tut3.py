from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize model
model = init_chat_model("llama-3.1-8b-instant", model_provider="groq", temperature=0.3)

# Define prompt template
template = """translate the given text
text: {context}
language: {question}
answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Create chain
chain = LLMChain(llm=model, prompt=prompt)

# Loop for input
while True:
    input_text = input("Enter the text: ")
    input_question = input("Enter the language: ")

    # Run the chain
    response = chain.run({
        "context": input_text,
        "question": input_question
    })
    
    print("Translated text:", response)
