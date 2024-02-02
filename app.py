from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

llm_open_ai = OpenAI()
llm_hugging_face = HuggingFaceHub(
    repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.9, "max_length": 64}
)

template = """Question: {question}"""
prompt = PromptTemplate(template=template, input_variables=["question"])

def get_openai_answer(question):
    """Get the answer using OpenAI language model."""
    llm_chain = LLMChain(prompt=prompt, llm=llm_open_ai)
    answer = llm_chain.invoke(question)
    return answer

def get_huggingface_answer(question):
    """Get the answer using Hugging Face language model."""
    llm_chain = LLMChain(prompt=prompt, llm=llm_hugging_face)
    answer = llm_chain.invoke(question)
    return answer

def get_text():
    """Get user input text."""
    return st.text_input("You: ", key="input")

st.set_page_config(page_title="LangChain QA", page_icon="🔗", layout="centered", initial_sidebar_state="expanded")
st.header("LangChain QA")

selected_option = st.radio(
    'How would you use?',
    ('OpenAi', 'HuggingFace'))

user_input = get_text()

try:
    if selected_option == 'OpenAi':
        response = get_openai_answer(user_input)
    else:
        response = get_huggingface_answer(user_input)
except Exception as e:
    st.error(f"Error: {str(e)}")
else:
    submit = st.button("Ask")

    if submit:
        st.subheader("Answer:")
        st.write(response["text"])
        st.balloons()
