from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print("API Key:", api_key)  
model = init_chat_model("llama-3.1-8b-instant", model_provider="groq",temperature=0.3)
template="what is the capital {city}" 
prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=model, prompt=prompt)
cities=["india","america","france","germany"]
for city in cities:
    output=chain.run(city)
    print(output)

