# RAG-Based LLM Application 


A Retrieval-Augmented Generation (RAG) application that enhances Large Language Model (LLM) responses by incorporating external knowledge sources. This system combines efficient document indexing, semantic vector search, re-ranking of retrieved results, and a user-friendly interface to provide accurate and context-aware answers.

Built with **LLaMA 3.2**, this application runs fully locally, leveraging modern tools such as Ollama, ChromaDB, and SentenceTransformers. 

---

## ⚙️ Prerequisites

Before running the application, ensure the following models are downloaded:

### 🔸 Ollama Models
Used for local inference and text embedding:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

- `nomic-embed-text` – used for generating semantic embeddings.
- `llama3.2:3b` – the primary LLM used for generating answers.

### 🔸 Re-Ranking Model
The re-ranking step uses the SentenceTransformer model `ms-marco-MiniLM-L6-v2`. You need to download it manually.

---

## 📦 Project Dependencies

This project uses several key libraries:

### 1. **Ollama** – Local Inference  
Runs LLMs locally, removing the need for cloud-based APIs.

### 2. **ChromaDB** – Vector Database  
Stores and retrieves embeddings for semantic document search.

### 3. **SentenceTransformers** – Re-Ranking  
Reorders vector search results based on contextual relevance.

### 4. **Streamlit** – User Interface  
Provides a simple and interactive UI for interacting with the LLM.

### 5. **PyMuPDF** – PDF Processing  
Extracts text from PDF files for ingestion and indexing.

### 6. **Langchain-Community** – Utilities & Integration  
Supports integration with LLMs and retrieval pipelines.

---

## 🚀 Installing Dependencies

Install all required libraries with:

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install ollama chromadb sentence-transformers streamlit pymupdf langchain-community
```

---
