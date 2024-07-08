import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering", model="google/flan-t5-base")

def fetch_website_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def answer_question(context, question):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

st.title("Advanced Website Q&A Chatbot")

url = st.text_input("Enter website URL:")
question = st.text_input("Ask a question about the website:")

if url and question:
    with st.spinner("Fetching website content..."):
        content = fetch_website_content(url)
    
    with st.spinner("Analyzing and answering..."):
        answer = answer_question(content, question)
    
    st.write("Answer:", answer)

st.markdown("---")
st.write("Chat History:")
for i, (q, a) in enumerate(st.session_state.get('chat_history', []), 1):
    st.write(f"Q{i}: {q}")
    st.write(f"A{i}: {a}")

if url and question:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append((question, answer))

