import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize the model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def fetch_website_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def answer_question(context, question):
    input_text = f"question: {question}\ncontext: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_return_sequences=1)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

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
