from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
load_dotenv()
api_key=os.getenv("GROQ_API_KEY")
prompt=PromptTemplate.from_template("what is the e commerce store that sells this {product}")
llm=init_chat_model(temperature=0.3, model_provider="groq", model="llama3-8b-8192")
chain=LLMChain(llm=llm, prompt=prompt)
product='IPHONE 13 in india'
output=chain.run(product)
print(output)