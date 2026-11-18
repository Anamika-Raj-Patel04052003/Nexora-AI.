
"""
NEXORA AI - RAG Pipeline

Handles document ingestion, chunking, vector storage
and conversation-specific document retrieval.
"""

import logging

from pathlib import Path

from app.rag.loaders import load_document
from app.rag.splitters import split_documents
from app.rag.embeddings import get_embedding_model

from app.rag.vector_store import add_documents_to_vector_store
from app.rag.retriever import retrieve_documents


logger = logging.getLogger(__name__)


# ==========================================
# 1. INGEST DOCUMENT
# ==========================================

def ingest_document(
    file_path: str,
    conversation_id: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> dict:
    """
    Load, split, embed and store a document.
    """

    # --------------------------------------
    # INPUT VALIDATION
    # --------------------------------------

    if not isinstance(conversation_id, int) or isinstance(conversation_id, bool) or conversation_id < 1:
        raise ValueError("Invalid conversation ID.")

    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("Valid file path is required.")

    document_path = Path(file_path)

    if not document_path.is_file():
        raise FileNotFoundError("Document file does not exist.")

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("Chunk size must be a positive integer.")

    if (
        not isinstance(chunk_overlap, int)
        or isinstance(chunk_overlap, bool)
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    # --------------------------------------
    # LOAD DOCUMENT
    # --------------------------------------

    documents = load_document(str(document_path))

    if not documents:
        raise ValueError("No readable content found in document.")

    # --------------------------------------
    # SPLIT DOCUMENT
    # --------------------------------------

    chunks = split_documents(
        documents,
        chunk_size,
        chunk_overlap
    )

    if not chunks:
        raise ValueError("No chunks generated from document.")

    # --------------------------------------
    # ADD METADATA
    # --------------------------------------

    for chunk in chunks:
        chunk.metadata["conversation_id"] = conversation_id
        chunk.metadata["filename"] = document_path.name

    # --------------------------------------
    # INITIALIZE EMBEDDINGS
    # --------------------------------------

    get_embedding_model()

    # --------------------------------------
    # STORE IN VECTOR DATABASE
    # --------------------------------------

    ids = add_documents_to_vector_store(chunks)

    if ids is None or len(ids) != len(chunks):
        raise RuntimeError(
            "Some document chunks may not have been stored successfully."
        )

    logger.info(
        "Document ingested successfully. File: %s, Conversation: %s, Chunks: %s",
        document_path.name,
        conversation_id,
        len(chunks)
    )

    return {
        "success": True,
        "filename": document_path.name,
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
    """
    Retrieve relevant document chunks
    belonging to a specific conversation.
    """

    # --------------------------------------
    # INPUT VALIDATION
    # --------------------------------------

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Search question cannot be empty.")

    if (
        not isinstance(conversation_id, int)
        or isinstance(conversation_id, bool)
        or conversation_id < 1
    ):
        raise ValueError("Valid conversation ID is required.")

    if (
        not isinstance(search_k, int)
        or isinstance(search_k, bool)
        or search_k < 1
        or search_k > 20
    ):
        raise ValueError("Search K must be between 1 and 20.")

    # --------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------

    documents = retrieve_documents(
        question=question.strip(),
        search_k=search_k,
        conversation_id=conversation_id
    )

    logger.info(
        "Document retrieval completed. Conversation: %s, Results: %s",
        conversation_id,
        len(documents) if documents else 0
    )

    return documents


# ==========================================
# 3. PROCESS DOCUMENT
# ==========================================

def process_document(
    file_path: str,
    conversation_id: int
) -> dict:
    """
    Main entry point for document processing.
    """

    return ingest_document(
        file_path=file_path,
        conversation_id=conversation_id
    )
