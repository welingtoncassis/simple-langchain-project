from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_openai import OpenAI

load_dotenv()
# print(os.environ["OPENAI_API_KEY"])

llm = OpenAI()

template = """Question: {question}"""
our_query = "What is the currency of India?"

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

completion = llm_chain.invoke(our_query)

print(completion)
