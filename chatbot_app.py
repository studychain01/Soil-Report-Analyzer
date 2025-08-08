import os
import streamlit as st
from openai import OpenAI

# Initialize OpenAI client using API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("LLM Chatbot")

# Keep chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Send a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the LLM
    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
        )
        reply = response.choices[0].message.content
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
