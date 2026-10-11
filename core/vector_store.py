import os
import threading
import time
import uuid
 
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
 
CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
 
_embeddings = None
_lock = threading.Lock()
 
 
def get_embeddings():
    """Load the embedding model ONCE per process and reuse it.
    (The old version created a new HuggingFaceEmbeddings, i.e. reloaded the model, on every call.)"""
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                t0 = time.perf_counter()
                print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
                _embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu"},
                )
                print(f"Embedding model ready in {time.perf_counter() - t0:.1f}s")
    return _embeddings
 
 
def warm_up_embeddings():
    """Start loading the embedding model in a background thread so the first
    analysis does not have to wait for it."""
    threading.Thread(target=get_embeddings, daemon=True).start()
 
 
def build_vector_store(transcript: str, persist: bool = False) -> Chroma:
    """Index one transcript.
 
    persist=False (default): fresh in-memory index with a unique collection name,
    so chat answers only ever come from the CURRENT video.
    persist=True: save to CHROMA_DIR, replacing whatever was stored before.
    """
    print("Building vector store")
    t0 = time.perf_counter()
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)
 
    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
 
    embeddings = get_embeddings()
    t_embed = time.perf_counter()
 
    if persist:
        try:  # drop the old transcript so the collection never mixes videos
            Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_DIR,
            ).delete_collection()
        except Exception:
            pass
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )
    else:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=f"{COLLECTION_NAME}_{uuid.uuid4().hex[:8]}",
        )
 
    print(f"Indexed {len(docs)} chunks (model wait {t_embed - t0:.1f}s, "
          f"indexing {time.perf_counter() - t_embed:.1f}s)")
    return vector_store
 
 
def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
 
    return vector_store
 
 
def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )