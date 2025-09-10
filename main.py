import os
from langchain_groq import ChatGroq

# Set API key (for quick test, but better to set it in environment)
os.environ["GROQ_API_KEY"] = "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"

# Initialize Groq model
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# Invoke the model
response = model.invoke("ask question to test the iq of a human")
print(response.content)
