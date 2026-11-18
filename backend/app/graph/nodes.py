
"""
NEXORA AI - LangGraph Nodes

Handles:
- Input validation
- Intent classification
- Normal conversational chat
- Document-based RAG
- Response and conversation history

CRAG and Self-RAG nodes will be integrated
with the retrieval grader and web search services.
"""

import logging

from langchain_core.messages import HumanMessage, AIMessage

from app.chains.chat_chain import generate_chat_response
from app.rag.rag_chain import answer_question


logger = logging.getLogger(__name__)


# ==========================================
# NODE 1: INPUT NODE
# ==========================================

def input_node(state: dict) -> dict:

    user_input = state.get("user_input", "")

    if not isinstance(user_input, str) or not user_input.strip():
        return {
            "error": "User input cannot be empty."
        }

    return {
        "user_input": user_input.strip(),
        "query": user_input.strip(),
        "error": ""
    }


# ==========================================
# NODE 2: INTENT CLASSIFICATION NODE
# ==========================================

def intent_node(state: dict) -> dict:

    existing_intent = state.get("intent", "auto")

    # Explicit RAG request
    if existing_intent == "rag":
        return {
            "intent": "rag"
        }

    # Explicit normal chat request
    if existing_intent == "chat":
        return {
            "intent": "chat"
        }

    # Auto mode:
    # A conversation ID allows document retrieval.
    # The retrieval result will determine whether
    # the workflow should use documents or normal chat.
    if state.get("conversation_id") is not None:
        return {
            "intent": "rag"
        }

    return {
        "intent": "chat"
    }


# ==========================================
# NODE 3: CHAT NODE
# ==========================================

def chat_node(state: dict) -> dict:

    user_input = state.get("user_input", "")

    history = state.get("messages", [])

    if not user_input:
        return {
            "error": "Chat input is missing."
        }

    try:

        response = generate_chat_response(
            message=user_input,
            history=history
        )

        return {
            "response": response,
            "answer": response,
            "answer_status": "generated",
            "response_type": "chat"
        }

    except Exception as error:

        logger.exception("Chat node failed.")

        return {
            "error": "Unable to generate chat response."
        }


# ==========================================
# NODE 4: DOCUMENT RAG NODE
# ==========================================

def rag_node(state: dict) -> dict:

    user_input = state.get("user_input", "")

    conversation_id = state.get("conversation_id")

    if conversation_id is None:
        return {
            "error": "Conversation ID is required for document retrieval."
        }

    try:

        result = answer_question(
            question=user_input,
            conversation_id=conversation_id
        )

        answer = result.get("answer", "")
        context = result.get("context", "")
        sources = result.get("sources", [])

        # No relevant document context was found.
        if not context.strip():

            return {
                "retrieved_documents": [],
                "retrieved_context": "",
                "sources": [],
                "retrieval_status": "no_relevant_documents",
                "needs_web_search": True,
                "response": "",
                "answer": ""
            }

        return {
            "retrieved_context": context,
            "sources": sources,
            "retrieval_status": "documents_found",
            "needs_web_search": False,
            "response": answer,
            "answer": answer,
            "answer_status": "generated",
            "response_type": "rag"
        }

    except Exception:

        logger.exception("Document RAG node failed.")

        return {
            "error": "Document retrieval or answer generation failed."
        }


# ==========================================
# NODE 5: RESPONSE NODE
# ==========================================

def response_node(state: dict) -> dict:

    user_input = state.get("user_input", "")

    response = (
        state.get("final_answer")
        or state.get("response")
        or state.get("answer")
        or ""
    )

    if not response.strip():

        return {
            "error": "No response was generated."
        }

    new_messages = [
        HumanMessage(content=user_input),
        AIMessage(content=response)
    ]

    return {
        "messages": new_messages,
        "response": response,
        "final_answer": response
    }
