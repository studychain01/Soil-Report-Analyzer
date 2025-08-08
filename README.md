# Soil Report Analyzer

This repository now includes a simple chatbot built with Streamlit and the OpenAI API.    

It supports Retrieval-Augmented Generation (RAG) by letting you upload a PDF file
which is chunked, embedded, and indexed with a small FAISS database for context-aware answers.


## Running the Chatbot

1. Install the dependencies:

```
pip install -r requirements.txt
```

2. Set your OpenAI API key and launch the Streamlit app:

```
OPENAI_API_KEY="your_api_key" streamlit run chatbot_app.py
```

          
3. Upload a PDF through the uploader in the app to build a FAISS index of the document.
   Questions you ask will retrieve relevant chunks from the PDF to help the model answer.

The app will open in your browser and allow you to chat with the model, using the
uploaded PDF for additional context when available.

