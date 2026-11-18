"""
NEXORA AI - LangGraph Workflow Orchestration

Assembles all nodes and conditional edges into a persistent StateGraph:
- Checkpointing with SQLite (nexora_checkpoints.db)
- Supervisor Agent multimodal routing
- In-chat Image, Audio, Video, and Blog generation
- CRAG & Self-RAG end-to-end orchestration
- Unified streaming support for chat, document RAG, and media answers
"""

import logging
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
    retrieve_documents_node,
    grade_documents_node,
    rewrite_query_node,
    web_search_node,
    generate_answer_node,
    evaluate_answer_node,
    revise_answer_node,
    generate_image_node,
    generate_audio_node,
    generate_video_node,
    generate_blog_node,
    response_node
)
from app.graph.edges import (
    route_after_input,
    route_by_intent,
    route_after_retrieval,
    route_after_document_grading,
    route_after_query_rewrite,
    route_after_web_search,
    route_after_generation,
    route_after_evaluation,
    route_after_revision,
    route_after_response
)


logger = logging.getLogger(__name__)

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
checkpointer.setup()


# ==========================================
# 2. INITIALIZE STATE GRAPH
# ==========================================

workflow = StateGraph(NexoraState)


# ==========================================
# 3. REGISTER ALL NODES
# ==========================================

# Core nodes
workflow.add_node("input", input_node)
workflow.add_node("intent", intent_node)
workflow.add_node("chat", chat_node)

# RAG & CRAG nodes
workflow.add_node("retrieve_documents", retrieve_documents_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("evaluate_answer", evaluate_answer_node)
workflow.add_node("revise_answer", revise_answer_node)

# Multimodal in-chat generation nodes
workflow.add_node("generate_image", generate_image_node)
workflow.add_node("generate_audio", generate_audio_node)
workflow.add_node("generate_video", generate_video_node)
workflow.add_node("generate_blog", generate_blog_node)

# Output node
workflow.add_node("response", response_node)


# ==========================================
# 4. CONNECT EDGES & CONDITIONAL ROUTING
# ==========================================

# START -> input
workflow.add_edge(START, "input")

# input -> intent or END
workflow.add_conditional_edges(
    "input",
    route_after_input,
    {
        "intent": "intent",
        END: END
    }
)

# intent -> routes to chat, RAG, or Multimodal generation
workflow.add_conditional_edges(
    "intent",
    route_by_intent,
    {
        "chat": "chat",
        "retrieve_documents": "retrieve_documents",
        "generate_image": "generate_image",
        "generate_audio": "generate_audio",
        "generate_video": "generate_video",
        "generate_blog": "generate_blog"
    }
)

# Multimodal generators connect to response
workflow.add_edge("generate_image", "response")
workflow.add_edge("generate_audio", "response")
workflow.add_edge("generate_video", "response")
workflow.add_edge("generate_blog", "response")

# chat -> response
workflow.add_edge("chat", "response")

# retrieve_documents -> grade_documents or web_search
workflow.add_conditional_edges(
    "retrieve_documents",
    route_after_retrieval,
    {
        "grade_documents": "grade_documents",
        "web_search": "web_search"
    }
)

# grade_documents -> generate_answer or rewrite_query or web_search
workflow.add_conditional_edges(
    "grade_documents",
    route_after_document_grading,
    {
        "generate_answer": "generate_answer",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search"
    }
)

# rewrite_query -> retrieve_documents or web_search
workflow.add_conditional_edges(
    "rewrite_query",
    route_after_query_rewrite,
    {
        "retrieve_documents": "retrieve_documents",
        "web_search": "web_search"
    }
)

# web_search -> generate_answer
workflow.add_conditional_edges(
    "web_search",
    route_after_web_search,
    {
        "generate_answer": "generate_answer"
    }
)

# generate_answer -> evaluate_answer
workflow.add_conditional_edges(
    "generate_answer",
    route_after_generation,
    {
        "evaluate_answer": "evaluate_answer"
    }
)

# evaluate_answer -> response or revise_answer
workflow.add_conditional_edges(
    "evaluate_answer",
    route_after_evaluation,
    {
        "response": "response",
        "revise_answer": "revise_answer"
    }
)

# revise_answer -> evaluate_answer
workflow.add_conditional_edges(
    "revise_answer",
    route_after_revision,
    {
        "evaluate_answer": "evaluate_answer"
    }
)

# response -> END
workflow.add_edge("response", END)


# ==========================================
# 5. COMPILE GRAPH WITH CHECKPOINTER
# ==========================================

nexora_graph = workflow.compile(
    checkpointer=checkpointer
)


# ==========================================
# 6. RUN WORKFLOW (INVOKE)
# ==========================================

def run_nexora(
    user_input: str,
    intent: str = "auto",
    conversation_id: int | str | None = None,
    file_path: str | None = None
) -> dict:
    thread_id = (
        str(conversation_id)
        if conversation_id is not None
        else str(uuid.uuid4())
    )

    initial_state = {
        "user_input": user_input,
        "query": user_input,
        "intent": intent,
        "conversation_id": conversation_id,
        "file_path": file_path,
        "messages": [],
        "retrieved_documents": [],
        "relevant_documents": [],
        "irrelevant_documents": [],
        "retrieved_context": "",
        "web_results": [],
        "web_context": "",
        "web_sources": [],
        "sources": [],
        "answer": "",
        "final_answer": "",
        "response": "",
        "retry_count": 0,
        "max_retries": 2,
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
# 7. STREAM WORKFLOW
# ==========================================

def stream_nexora(
    user_input: str,
    conversation_id: int,
    intent: str = "auto",
    file_path: str | None = None
):
    final_output = run_nexora(
        user_input=user_input,
        intent=intent,
        conversation_id=conversation_id,
        file_path=file_path
    )

    final_text = (
        final_output.get("final_answer")
        or final_output.get("response")
        or final_output.get("answer")
        or ""
    )

    if final_text:
        words = final_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
    else:
        err = final_output.get("error", "No response could be generated.")
        yield f"Error: {err}"