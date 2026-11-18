
"""
NEXORA AI - LangGraph State

Maintains conversation, document retrieval, web search,
CRAG decisions and Self-RAG evaluation information.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class NexoraState(TypedDict, total=False):

    # Conversation
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    query: str
    rewritten_query: str
    intent: str

    # Conversation information
    conversation_id: int | str
    file_path: str

    # Document retrieval
    retrieved_documents: list[Document]
    relevant_documents: list[Document]
    irrelevant_documents: list[Document]
    retrieved_context: str
    sources: list[dict[str, Any]]

    retrieval_status: str
    retrieval_score: float
    retrieval_attempts: int

    # CRAG
    document_relevance: str
    retrieval_decision: str
    needs_web_search: bool

    # Web search
    web_search_query: str
    web_results: list[dict[str, Any]]
    web_context: str
    web_sources: list[dict[str, Any]]

    # Answer generation
    answer: str
    response: str

    # Self-RAG
    answer_relevant: bool
    answer_grounded: bool
    evaluation_feedback: str
    evaluation_score: float
    evaluation_result: dict[str, Any]

    retry_count: int
    max_retries: int

    # Final response
    final_answer: str
    answer_status: str
    response_type: str

    # Error handling
    error: str
