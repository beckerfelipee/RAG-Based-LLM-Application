# RAG-Based-LLM-Application
A Retrieval-Augmented Generation (RAG) application that enhances LLM responses with external knowledge retrieval. Features efficient document indexing, vector search, and seamless integration with llama 3.2 for accurate and context-aware answers. nference, vector database storage, and intelligent ranking to improve response accuracy and relevance.

# Project Dependencies
This project utilizes various libraries for essential functionalities. Below is a description of each dependency and its purpose:

## 1. Ollama – Local Inference
Used for running AI model inference locally without relying on cloud services.

Enables efficient execution of language models directly on the user's machine.

## 2. ChromaDB – Vector Database
A vector database for storing and retrieving embeddings efficiently.

Essential for semantic search and similarity-based information retrieval.

## 3. Sentence-Transformers – Re-Ranking
Used to reorder the results retrieved from the vector database.

Improves response accuracy by ranking the most relevant results higher.

## 4. Streamlit – User Interface (UI)
A framework for building interactive and user-friendly interfaces for Python applications.

Simplifies data visualization and interaction with the AI model.

## 5. PyMuPDF – PDF Processing
A library for handling and extracting text from PDF files.

Used for document preprocessing and data ingestion.

## 6. Langchain-Community – Auxiliary Tools
A collection of LangChain utilities that facilitate integration with LLMs and vector databases.

Used for specific components within the project's workflow.

## Installing Dependencies
To install all required libraries, run:

```bash
pip install -r requirements.txt
```
Or install them individually:
```bash
pip install ollama chromadb sentence-transformers streamlit pymupdf langchain-community
```
