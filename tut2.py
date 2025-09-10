from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


model = init_chat_model("llama-3.1-8b-instant", model_provider="groq", temperature=0.3)

# Prompt template
template = """You are an assistant for a Q&A task.
Use the following context to answer the question.
If you don't know, say "I don't know the answer."
Keep your response within 3 sentences.

Context: {context}
Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


chain = LLMChain(llm=model, prompt=prompt)

# Read book safely (UTF-8 encoding)
with open("book.txt", "r", encoding="utf-8") as f:
    book_context = f.read()


chunk = book_context[:len(book_context)//40]  


response = chain.run({
    "context": chunk,
    "question": "what is the book about"
})

print(response) 
