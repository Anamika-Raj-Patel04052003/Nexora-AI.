
"""
NEXORA AI - RAG Agent

Retrieves relevant document chunks, formats context,
generates answers and extracts source information.

CRAG and Self-RAG decisions will be handled by LangGraph.
"""

import logging

from app.rag.retriever import retrieve_documents

from app.chains.rag_chain import (
    format_documents,
    generate_rag_answer
)


logger = logging.getLogger(__name__)


# --------------------------------------------------
# EXTRACT DOCUMENT SOURCES
# --------------------------------------------------

def extract_sources(documents: list) -> list:

    sources = []

    for document in documents:

        metadata = document.metadata or {}

        source_info = {
            "source": metadata.get(
                "source",
                "Unknown"
            ),

            "filename": metadata.get(
                "filename",
                "Unknown"
            ),

            "page": metadata.get(
                "page",
                "Unknown"
            )
        }

        if source_info not in sources:
            sources.append(source_info)

    return sources


# --------------------------------------------------
# ASK QUESTION USING DOCUMENT RAG
# --------------------------------------------------

def answer_question(
    question: str,
    conversation_id: int
) -> dict:

    # Validate question
    if not isinstance(question, str) or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    # Validate conversation ID
    if (
        not isinstance(conversation_id, int)
        or isinstance(conversation_id, bool)
        or conversation_id < 1
    ):

        raise ValueError(
            "Valid conversation ID is required."
        )

    logger.info(
        "Processing document RAG question for conversation %s.",
        conversation_id
    )

    # --------------------------------------
    # 1. RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------

    documents = retrieve_documents(
        question=question,
        search_k=5,
        conversation_id=conversation_id
    )

    # --------------------------------------
    # 2. HANDLE EMPTY DOCUMENTS
    # --------------------------------------

    if not documents:

        logger.info(
            "No document chunks retrieved for conversation %s.",
            conversation_id
        )

        return {
            "answer": (
                "I could not find relevant information "
                "in the uploaded documents."
            ),
            "context": "",
            "sources": [],
            "retrieval_status": "no_documents",
            "retrieved_count": 0
        }

    # --------------------------------------
    # 3. FORMAT DOCUMENT CONTEXT
    # --------------------------------------

    context = format_documents(documents)

    if not context or not context.strip():

        logger.warning(
            "Retrieved documents produced empty context."
        )

        return {
            "answer": (
                "I could not extract readable context "
                "from the retrieved documents."
            ),
            "context": "",
            "sources": [],
            "retrieval_status": "empty_context",
            "retrieved_count": len(documents)
        }

    # --------------------------------------
    # 4. GENERATE DOCUMENT-BASED ANSWER
    # --------------------------------------

    answer = generate_rag_answer(
        context=context,
        question=question
    )

    if not isinstance(answer, str) or not answer.strip():

        raise RuntimeError(
            "The RAG model returned an empty answer."
        )

    # --------------------------------------
    # 5. EXTRACT SOURCE INFORMATION
    # --------------------------------------

    sources = extract_sources(documents)

    # --------------------------------------
    # 6. RETURN COMPLETE RESULT
    # --------------------------------------

    logger.info(
        "Document RAG answer generated using %d retrieved chunks.",
        len(documents)
    )

    return {
        "answer": answer,
        "context": context,
        "sources": sources,
        "retrieval_status": "documents_retrieved",
        "retrieved_count": len(documents)
    }
