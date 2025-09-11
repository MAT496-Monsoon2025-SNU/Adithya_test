# %%
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()  # loads your GROQ_API_KEY from .env

from langchain_unstructured import UnstructuredLoader
from langchain_groq import ChatGroq

# Load webpage
loader = UnstructuredLoader(web_url="https://maths.du.ac.in/faculty-profile/")
docs = loader.load()

full_doc = "\n\n".join(doc.page_content for doc in docs)
print("Document length:", len(full_doc))

# Prompt template
prompt_template = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say you don't know.
Keep the answer concise.

Question: {question}
Context: {context}
Answer:"""

question = "Make a list of all email addresses."

# ✅ Use supported Groq model here
llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq")


response = llm.invoke(
    prompt_template.format(context=full_doc, question=question)
)

print("\n=== Extracted Email Addresses ===\n")
print(response.content)
