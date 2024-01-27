from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_community.llms import HuggingFaceHub

load_dotenv()


repo_id = "google/flan-t5-large"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.9, "max_length": 64}
)

template = """Question: {question}"""
our_query = "What is the currency of India?"

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

completion = llm_chain.invoke(our_query)

print(completion)
