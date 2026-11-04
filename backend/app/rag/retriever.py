
from app.rag.vector_store import get_vector_store


# ==========================================
# 1. GET DOCUMENT RETRIEVER
# ==========================================

def get_document_retriever(
    conversation_id: int,
    search_k: int = 5
):

    if conversation_id < 1:
        raise ValueError("Invalid conversation ID.")

    if search_k < 1:
        raise ValueError(
            "search_k must be at least 1."
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
):

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if conversation_id is None or conversation_id < 1:
        raise ValueError(
            "Valid conversation ID is required."
        )

    retriever = get_document_retriever(
        conversation_id=conversation_id,
        search_k=search_k
    )

    documents = retriever.invoke(
        question
    )

    return documents
