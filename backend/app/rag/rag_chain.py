
from app.rag.retriever import retrieve_documents

from app.chains.rag_chain import (
    format_documents,
    generate_rag_answer
)


# ==========================================
# 1. ASK QUESTION USING RAG
# ==========================================

def answer_question(
    question: str,
    conversation_id: int
):

    # Validate question
    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    # Validate conversation ID
    if conversation_id is None or conversation_id < 1:

        raise ValueError(
            "Valid conversation ID is required."
        )

    # --------------------------------------
    # 2. RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------

    documents = retrieve_documents(
        question=question,
        search_k=5,
        conversation_id=conversation_id
    )

    # --------------------------------------
    # 3. HANDLE EMPTY DOCUMENTS
    # --------------------------------------

    if not documents:

        return {
            "answer": (
                "I could not find relevant information "
                "in the uploaded documents."
            ),
            "context": "",
            "sources": []
        }

    # --------------------------------------
    # 4. FORMAT DOCUMENT CONTEXT
    # --------------------------------------

    context = format_documents(
        documents
    )

    # --------------------------------------
    # 5. GENERATE ANSWER
    # --------------------------------------

    answer = generate_rag_answer(
        context=context,
        question=question
    )

    # --------------------------------------
    # 6. EXTRACT SOURCE INFORMATION
    # --------------------------------------

    sources = []

    for document in documents:

        source_info = {
            "source": document.metadata.get(
                "source",
                "Unknown"
            ),

            "page": document.metadata.get(
                "page",
                "Unknown"
            )
        }

        if source_info not in sources:

            sources.append(source_info)

    # --------------------------------------
    # 7. RETURN COMPLETE RESULT
    # --------------------------------------

    return {
        "answer": answer,
        "context": context,
        "sources": sources
    }
