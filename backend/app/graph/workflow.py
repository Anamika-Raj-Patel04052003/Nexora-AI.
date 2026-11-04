
import sqlite3
import uuid

from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph.state import NexoraState

from app.graph.nodes import (
    input_node,
    intent_node,
    chat_node,
    rag_node,
    response_node
)

from app.graph.edges import (
    route_after_input,
    route_by_intent
)


# ==========================================
# 1. SQLITE CHECKPOINTER SETUP
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

CHECKPOINT_DB = BASE_DIR / "nexora_checkpoints.db"

checkpoint_connection = sqlite3.connect(
    str(CHECKPOINT_DB),
    check_same_thread=False
)

checkpointer = SqliteSaver(checkpoint_connection)

# Create checkpoint tables if they do not exist
checkpointer.setup()


# ==========================================
# 2. INITIALIZE STATE GRAPH
# ==========================================

workflow = StateGraph(NexoraState)


# ==========================================
# 3. REGISTER NODES
# ==========================================

workflow.add_node("input", input_node)
workflow.add_node("intent", intent_node)
workflow.add_node("chat", chat_node)
workflow.add_node("rag", rag_node)
workflow.add_node("response", response_node)


# ==========================================
# 4. CONNECT START EDGE
# ==========================================

workflow.add_edge(START, "input")


# ==========================================
# 5. CONDITIONAL EDGE AFTER INPUT
# ==========================================

workflow.add_conditional_edges(
    "input",
    route_after_input,
    {
        "intent": "intent",
        END: END
    }
)


# ==========================================
# 6. CONDITIONAL EDGE AFTER INTENT
# ==========================================

workflow.add_conditional_edges(
    "intent",
    route_by_intent,
    {
        "chat": "chat",
        "rag": "rag"
    }
)


# ==========================================
# 7. CHAT AND RAG CONNECTION
# ==========================================

workflow.add_edge("chat", "response")
workflow.add_edge("rag", "response")


# ==========================================
# 8. RESPONSE TO END
# ==========================================

workflow.add_edge("response", END)


# ==========================================
# 9. COMPILE WITH CHECKPOINTER
# ==========================================

nexora_graph = workflow.compile(
    checkpointer=checkpointer
)


# ==========================================
# 10. NORMAL WORKFLOW EXECUTION
# ==========================================

def run_nexora(
    user_input: str,
    intent: str = "chat",
    conversation_id: int | str | None = None,
    file_path: str | None = None
):

    thread_id = (
        str(conversation_id)
        if conversation_id is not None
        else str(uuid.uuid4())
    )

    initial_state = {
        "user_input": user_input,
        "intent": intent,
        "conversation_id": conversation_id,
        "file_path": file_path,
        "messages": [],
        "response": "",
        "retrieved_context": "",
        "sources": [],
        "error": ""
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = nexora_graph.invoke(
        initial_state,
        config=config
    )

    return result


# ==========================================
# 11. STREAMING WORKFLOW EXECUTION
# ==========================================

def stream_nexora(
    user_input: str,
    conversation_id: int,
    intent: str = "chat",
    file_path: str | None = None
):
    """
    Execute LangGraph and stream Chat Node tokens.

    Conversation state is maintained using thread_id.

    Note:
    RAG streaming is not implemented yet.
    """

    thread_id = str(conversation_id)

    initial_state = {
        "user_input": user_input,
        "intent": intent,
        "conversation_id": conversation_id,
        "file_path": file_path,
        "messages": [],
        "response": "",
        "retrieved_context": "",
        "sources": [],
        "error": ""
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    for chunk, metadata in nexora_graph.stream(
        initial_state,
        config=config,
        stream_mode="messages"
    ):

        # Stream only Chat Node tokens
        if metadata.get("langgraph_node") != "chat":
            continue

        content = getattr(chunk, "content", "")

        if isinstance(content, str) and content:
            yield content
