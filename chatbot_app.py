import os
import streamlit as st
import numpy as np
import faiss
from PyPDF2 import PdfReader


# Initialize OpenAI client using API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("LLM Chatbot with PDF RAG")

# Session state for chat history and document index
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_index" not in st.session_state:
    st.session_state.doc_index = None
    st.session_state.doc_chunks = []


def embed_text(text: str) -> np.ndarray:
    """Return embedding vector for given text using OpenAI embeddings."""
    response = client.embeddings.create(
        model="text-embedding-3-small", input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


def chunk_text(text: str, size: int = 200, overlap: int = 20):
    """Yield overlapping word chunks from text."""
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i : i + size])


def process_pdf(file) -> None:
    """Extract text from an uploaded PDF and build a FAISS index."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    chunks = list(chunk_text(text))
    if not chunks:
        return

    embeddings = np.vstack([embed_text(chunk) for chunk in chunks])
    index = faiss.IndexFlatL2(embeddings.shape[1])  # brute-force index
    index.add(embeddings)

    st.session_state.doc_index = index
    st.session_state.doc_chunks = chunks


def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve top-k relevant chunks for the query."""
    if st.session_state.doc_index is None:
        return ""

    query_vec = embed_text(query)
    distances, indices = st.session_state.doc_index.search(
        np.array([query_vec]), k
    )
    return "\n\n".join(
        st.session_state.doc_chunks[i]
        for i in indices[0]
        if i < len(st.session_state.doc_chunks)
    )


# PDF uploader to build RAG index
uploaded_file = st.file_uploader("Upload a PDF for context", type="pdf")
if uploaded_file is not None:
    process_pdf(uploaded_file)
    st.success("PDF processed and indexed.")



# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


    context = retrieve_context(prompt)
    messages = st.session_state.messages.copy()
    if context:
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"Use the following context to answer:\n{context}",
            },
        )

    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages

        )
        reply = response.choices[0].message.content
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

