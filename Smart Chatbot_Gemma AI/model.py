import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question:{question}")
    ]
)

# Streamlit UI
st.set_page_config(page_title="Smart Chatbot", layout="wide")
st.title("💬 Smart Chatbot with Gemma AI")
st.markdown("Ask any question and get instant answers from Gemma AI.")

# Input box with placeholder
input_text = st.text_input("Type your question here...", placeholder="e.g., What is LangChain?")

# Ollama Llama2 model
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Display output nicely
if input_text:
    st.markdown("---")  # horizontal line
    st.markdown("**Your Question:**")
    st.write(input_text)
    st.markdown("**Gemma AI Response:**")
    st.write(chain.invoke({"question":input_text}))
