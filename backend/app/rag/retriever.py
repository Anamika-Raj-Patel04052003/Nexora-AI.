
"""
NEXORA AI - Document Retriever

Retrieves relevant document chunks from PGVector
using conversation-specific semantic similarity search.
"""

import logging

from app.rag.vector_store import get_vector_store


logger = logging.getLogger(__name__)


# ==========================================
# 1. GET DOCUMENT RETRIEVER
# ==========================================

def get_document_retriever(
    conversation_id: int,
    search_k: int = 5
):
    """Create a conversation-specific document retriever."""

    if (
        not isinstance(conversation_id, int)
        or isinstance(conversation_id, bool)
        or conversation_id < 1
    ):
        raise ValueError("Invalid conversation ID.")

    if (
        not isinstance(search_k, int)
        or isinstance(search_k, bool)
        or search_k < 1
        or search_k > 20
    ):
        raise ValueError(
            "search_k must be between 1 and 20."
        )

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": search_k,
            "filter": {
                "conversation_id": conversation_id
            }
        }
    )

    return retriever


# ==========================================
# 2. RETRIEVE DOCUMENTS
# ==========================================

def retrieve_documents(
    question: str,
    search_k: int = 5,
    conversation_id: int | None = None
) -> list:
    """
    Retrieve relevant chunks belonging to
    the requested conversation.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if (
        not isinstance(conversation_id, int)
        or isinstance(conversation_id, bool)
        or conversation_id < 1
    ):
        raise ValueError(
            "Valid conversation ID is required."
        )

    retriever = get_document_retriever(
        conversation_id=conversation_id,
        search_k=search_k
    )

    documents = retriever.invoke(
        question.strip()
    )

    if not documents:
        logger.info(
            "No relevant documents found. Conversation: %s",
            conversation_id
        )

        return []

    logger.info(
        "Retrieved %s document chunks. Conversation: %s",
        len(documents),
        conversation_id
    )

    return documents
