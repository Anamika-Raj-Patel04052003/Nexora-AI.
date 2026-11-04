
from langchain_core.messages import HumanMessage, AIMessage

from app.chains.chat_chain import generate_chat_response
from app.rag.rag_chain import answer_question


# ==========================================
# NODE 1: INPUT NODE
# ==========================================

def input_node(state):

    user_input = state.get("user_input", "")

    if not user_input or not user_input.strip():
        return {
            "error": "User input cannot be empty."
        }

    return {
        "user_input": user_input.strip(),
        "error": ""
    }


# ==========================================
# NODE 2: INTENT CLASSIFICATION NODE
# ==========================================

def intent_node(state):

    existing_intent = state.get("intent")

    if existing_intent in ("chat", "rag"):
        intent = existing_intent

    elif state.get("retrieved_context") or state.get("file_path"):
        intent = "rag"

    else:
        intent = "chat"

    return {
        "intent": intent
    }


# ==========================================
# NODE 3: CHAT NODE
# ==========================================

def chat_node(state):

    user_input = state.get("user_input", "")

    history = state.get("messages", [])

    response = generate_chat_response(
        message=user_input,
        history=history
    )

    return {
        "response": response
    }


# ==========================================
# NODE 4: RAG NODE
# ==========================================

def rag_node(state):

    user_input = state.get("user_input", "")

    conversation_id = state.get("conversation_id")

    if conversation_id is None:
        return {
            "error": "Conversation ID is required for RAG."
        }

    # Generate answer using RAG chain
    result = answer_question(
        question=user_input,
        conversation_id=conversation_id
    )

    # Extract generated answer
    answer = result["answer"]

    # Extract retrieved document context
    context = result["context"]

    # Extract document sources
    sources = result["sources"]

    return {
        "response": answer,
        "retrieved_context": context,
        "sources": sources
    }


# ==========================================
# NODE 5: RESPONSE NODE
# ==========================================

def response_node(state):

    user_input = state.get("user_input", "")

    response = state.get("response", "")

    # Add user and assistant messages
    new_messages = [
        HumanMessage(content=user_input),
        AIMessage(content=response)
    ]

    return {
        "messages": new_messages,
        "response": response
    }
