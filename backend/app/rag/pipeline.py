
from app.rag.loaders import load_document
from app.rag.splitters import split_documents

from app.rag.embeddings import get_embedding_model

from app.rag.vector_store import (
    add_documents_to_vector_store
)

from app.rag.retriever import retrieve_documents


# ==========================================
# 1. INGEST DOCUMENT
# ==========================================

def ingest_document(
    file_path: str,
    conversation_id: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):

    if conversation_id < 1:
        raise ValueError("Invalid conversation ID.")

    # Load document
    documents = load_document(file_path)

    # Split document into chunks
    chunks = split_documents(
        documents,
        chunk_size,
        chunk_overlap
    )

    if not chunks:
        raise ValueError(
            "No chunks generated from document."
        )

    # Add conversation ID to every chunk
    for chunk in chunks:
        chunk.metadata["conversation_id"] = conversation_id

    # Initialize embedding model
    get_embedding_model()

    # Store chunks in PostgreSQL
    ids = add_documents_to_vector_store(chunks)

    return {
        "success": True,
        "filename": documents[0].metadata.get("source"),
        "conversation_id": conversation_id,
        "total_pages": len(documents),
        "total_chunks": len(chunks),
        "stored_vectors": len(ids)
    }


# ==========================================
# 2. SEARCH DOCUMENTS
# ==========================================

def search_documents(
    question: str,
    search_k: int = 5,
    conversation_id: int | None = None
):

    if conversation_id is None or conversation_id < 1:
        raise ValueError("Valid conversation ID is required.")

    documents = retrieve_documents(
        question=question,
        search_k=search_k,
        conversation_id=conversation_id
    )

    return documents


# ==========================================
# 3. PROCESS DOCUMENT
# ==========================================

def process_document(
    file_path: str,
    conversation_id: int
):

    return ingest_document(
        file_path=file_path,
        conversation_id=conversation_id
    )



