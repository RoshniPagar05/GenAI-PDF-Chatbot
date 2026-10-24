
# AI PDF Chatbot using RAG

## Overview
An AI-powered PDF question-answering application that
retrieves relevant passages from uploaded documents and
generates answers with source-page references.

## Features
- Upload PDF documents
- Extract and split PDF text into chunks
- Generate text embeddings
- Retrieve relevant passages using ChromaDB
- Ask questions in natural language
- Generate answers using a hosted LLM
- Display retrieved source passages and page numbers

## Technologies
- Python
- Streamlit
- LangChain Text Splitters
- Sentence Transformers
- ChromaDB
- Groq API
- PyPDF

## Installation
1. Clone or download this repository.
2. Create a virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Create `.streamlit/secrets.toml` and add:

   GROQ_API_KEY = "your_api_key"

5. Start the application:

   python -m streamlit run app.py

## How It Works
1. Extract text from the PDF.
2. Split the text into chunks.
3. Convert chunks into embeddings.
4. Retrieve relevant chunks for each question.
5. Send the retrieved context to the language model.
6. Display the generated answer and source pages.

## Limitations
- Scanned PDFs require OCR.
- Answer quality depends on PDF text quality and retrieval.
- API usage may be subject to provider limits.
- The current document index is stored in memory.

## Future Enhancements
- OCR support
- Multiple document collections
- Conversation memory
- Automated evaluation
- Improved source citations
