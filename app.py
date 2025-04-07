# ----- Dependencies ----- #

# OS
import os
import atexit
import psutil
import subprocess
import tempfile 

# UI
import streamlit as st

# Process document and split text
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

# get or create vector collection and query it
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# Reranking the results based on the question 
from sentence_transformers import CrossEncoder

# LLM model to generate the answer
import ollama

# Fix for torch attribute error
import torch
torch.classes.__path__ = [] 

# ----- Constants ----- #

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
RERANK_MODEL_PATH = "./ms-marco-MiniLM-L6-v2"
EMBEDDING_TOP_K = 10
RERANK_TOP_K = 3
VECTOR_DB_PATH = "./data-rag-chroma"

# ----- LLM usage ----- #

SYSTEM_PROMPT = """
You are a Capgemini AI assistant that answers **strictly based on the provided context**. Your goal is to analyze the context and generate responses **only from its content**.  

##  **Answering Rules:**  
1. **Only use the given context—no external knowledge or assumptions.**  
2. **If the context lacks relevant information, state that clearly.**  
3. **Ignore unrelated questions.**  

##  **Response Format:**  
- **Always reply in the users language.**  
- **Only include information from the context.**  
- **If information is missing, say so explicitly.**  
- Use structured responses with:  
  -  Clear paragraphs  
  -  Bullet points or lists when needed  

## ⚠ **Important:**  
**All provided context can be used freely.**  
**Never** answer questions that are unrelated to the given context. If the context does not provide enough information, **simply state that the question is not related.**  
"""

# ----- Functions ----- #

# Process the uploaded document and split it into smaller chunks

def process_document(uploaded_file: UploadedFile, isPrepared: bool = False) -> list[Document]:
    """Extract and split text from an uploaded PDF document."""
    print("Processing document...")
    
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    temp_file.close()
    os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    split_docs = text_splitter.split_documents(docs)
    return split_docs


def process_prepared_document(uploaded_file: UploadedFile, symbol: str) -> list[Document]:
    """Process pre-structured AI-ready documents using a custom delimiter."""
    print("Processing AI Ready document...")

    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    temp_file.close()
    os.unlink(temp_file.name)

    split_docs: list[Document] = []

    # For each page
    for doc in docs:
        page_text = doc.page_content
        page_metadata = doc.metadata

        chunks = page_text.split(symbol)

        # New document object for each chunk
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                split_docs.append(Document(page_content=chunk, metadata=page_metadata))

    return split_docs


# Ollama server management

def start_ollama():
    """Start the Ollama server if it's not already running."""
    print("Starting Ollama server...")
    try:
        with open(os.devnull, 'w') as devnull:
            subprocess.Popen(['ollama', 'serve'], stdout=devnull, stderr=devnull)
        print("Ollama server started.")
    except Exception as e:
        print(f"Failed to initiate Ollama: {e}")


def is_ollama_running() -> bool:
    """Check if the Ollama server is running."""
    print("Checking if Ollama server is running...")
    isRunning = any('ollama' in proc.info['name'].lower() for proc in psutil.process_iter(['name']))
    print(isRunning)
    return isRunning


def terminate_ollama_processes():
    """Terminate all running Ollama processes."""
    print("Terminating Ollama process...")
    for proc in psutil.process_iter(['pid', 'name']):
        if 'ollama' in proc.info['name'].lower():  
            proc.terminate() 
            proc.wait() 
            print(f"Ollama Process (PID {proc.info['pid']}) terminated")


# Vector collection management

def get_vector_collection() -> chromadb.Collection:
    """Retrieve or create a ChromaDB vector collection."""
    embedding_function = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name=EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH) # chroma uses sqlite3 to store data
    return chroma_client.get_or_create_collection(name="rag-app", embedding_function=embedding_function, metadata={"hnsw:space": "cosine"})
    # hnsw is a nearest neighbor search algorithm, cosine is a similarity measure.


def add_to_vector_collection(all_splits: list[Document], file_name: str) -> bool:
    """Add document chunks to the vector database."""
    collection = get_vector_collection()
    print("Adding documents to the collection...")

    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        page_number = split.metadata.get('page') + 1 # pages in metadata are 0-indexed
        split.page_content = f"Document: {file_name} Page: {page_number} {split.page_content}"
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_page_{page_number}_{idx}") # Create unique IDs for each chunk
    
    # create or update data in the collection
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Added {len(all_splits)} splits to the collection")
    return True


def query_collection(prompt: str, top_k: int = EMBEDDING_TOP_K):
    """Query the vector database based on a user prompt."""
    collection = get_vector_collection()
    print("Querying the collection...")
    return collection.query(query_texts=[prompt], n_results=top_k) if collection.count() > 0 else None


# Rerank the results using a cross-encoder model

def rerank_cross_encoders(prompt: str, documents: list[str], top_k: int = RERANK_TOP_K) -> tuple[str, list[int], list[dict]]:
    """Rerank retrieved documents using CrossEncoder."""
    print("Reranking...")
    relevant_text = ""
    relevant_text_ids = []

    model = CrossEncoder(RERANK_MODEL_PATH)
    ranks = model.rank(prompt, documents, top_k=top_k)
    for rank in ranks:
        relevant_text +=  documents[rank["corpus_id"]] + "\n _ _ _ _ \n"
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids, ranks


# Call the LLM model to generate the answer

def call_llm(context: str, question: str, model: str):
    """Call the LLM to generate an answer based on context."""
    print(f"Calling LLM model ({model})...")
    response = ollama.chat(
        model=model,
        stream=True, 
        messages=[
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": f"Question: {question}"
            },
            {
                "role": "user", 
                "content": f"Context: {context}"
            }
        ]
    )

    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]
        else:
            break


# ----- Main ----- #

if __name__ == "__main__":
    atexit.register(terminate_ollama_processes)
    st.set_page_config(page_title="RAG Interface", page_icon="🤖", layout="wide")
    print("\n---\n")

    st.markdown("""
        <h1 style='text-align: center; color: #0070AD;'>RAG Interface</h1>
        <p style='text-align: center; color: #0070AD;'>An Augmented Retrieval Generation solution for Internal Documents</p>
        <hr style='border: 1px solid #0070AD;'>
    """, unsafe_allow_html=True)

    if not is_ollama_running():
        with st.spinner("Starting the Ollama server..."):
            start_ollama()

    # Document upload and processing
    all_splits: list[Document] = []
    with st.sidebar:
        st.header("📂 Document Upload")
        uploaded_file = st.file_uploader("**Upload a PDF file**", type="pdf")

        if st.button("📄 Process Document"):
            if uploaded_file:
                with st.spinner("Processing the document..."):
                    file_name = uploaded_file.name.translate(str.maketrans({" ": "_", ".": "_", "-": "_", ",": "_", ";": "_", ":": "_"}))
                    all_splits = process_prepared_document(uploaded_file, "$") if file_name.startswith("RAG") else process_document(uploaded_file)
                    if add_to_vector_collection(all_splits, file_name):
                        st.success("✅ Document successfully added!")
                    else:
                        st.error("❌ Document processing failed.")
            else:
                st.warning("Please upload a PDF file to process.")

    st.header("💡 RAG Question Answering")
    prompt = st.text_area("**Ask a question related to the document:**").strip()

    if st.button("🤖 Ask"):
        if prompt:
            status = st.empty()

            # Embedding model query
            status.info("🔎 Querying the document collection...")
            results = query_collection(prompt) 
            if results:
                context = results.get("documents")[0]
                with st.popover("📜 Retrieved Documents"):
                    for n in range(len(results.get("documents")[0])):
                        st.write(f"**ID {n}:** {results.get('ids')[0][n]}")
                        st.write(results.get("documents")[0][n])
                        st.write(f"🔍 Distance: {results.get('distances')[0][n]}")
                        st.divider()
                
                # Rerank the results using cross-encoder Model
                status.info("⚖️ Reranking results...")
                relevant_text, relevant_text_ids, ranks = rerank_cross_encoders(prompt, context) 
                with st.popover("🏆 Most Relevant Document"):
                    st.write(ranks)
                    st.write(relevant_text)
                
                # Call the LLM model to generate the answer
                status.info("💬 Generating response...")
                response = call_llm(context=relevant_text, question=prompt, model=LLM_MODEL)
                st.write_stream(response)
                status.empty()
            else:
                status.error("❌ No relevant documents found.")
        else:
            st.warning("Please enter a question to ask.")
    
    if all_splits:    
        with st.popover("View processed document chunks", ):
            st.write(all_splits)
    