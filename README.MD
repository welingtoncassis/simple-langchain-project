# Simple LangChain Project

## Description

LangChain is a Python project that utilizes the LangChain library for natural language processing. In this example, we demonstrate how to use LangChain to interact with a Hugging Face an OpenAi language model to answer a specific question.

## Streamlit

I use `Streamlit` to create user interface, allowing them to enter questions and see the answer

## Configuration

- To run this code, make sure you have a .env file in your project directory with the following content:

```
# .env file
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_hugging_api_key
```

## Installation

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
