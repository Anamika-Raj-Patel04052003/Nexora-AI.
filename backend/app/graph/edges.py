
"""
NEXORA AI - LangGraph Conditional Edges

Controls routing between:
- Input validation
- Intent classification
- Document retrieval
- CRAG relevance grading
- Query rewriting
- Web search fallback
- Answer generation
- Self-RAG evaluation
- Final response
"""

from langgraph.graph import END


# ==========================================
# ROUTE 1: AFTER INPUT VALIDATION
# ==========================================

def route_after_input(state: dict):

    if state.get("error"):
        return END

    return "intent"


# ==========================================
# ROUTE 2: AFTER INTENT CLASSIFICATION
# ==========================================

def route_by_intent(state: dict):

    if state.get("error"):
        return END

    intent = state.get("intent", "chat")

    if intent == "rag":
        return "rag"

    return "chat"


# ==========================================
# ROUTE 3: AFTER DOCUMENT RETRIEVAL
# ==========================================

def route_after_retrieval(state: dict):

    if state.get("error"):
        return END

    if state.get("needs_web_search"):
        return "web_search"

    if not state.get("retrieved_documents"):
        return "web_search"

    return "grade_documents"


# ==========================================
# ROUTE 4: CRAG DOCUMENT GRADING
# ==========================================

def route_after_document_grading(state: dict):

    if state.get("error"):
        return END

    relevant_documents = state.get(
        "relevant_documents", []
    )

    if relevant_documents:
        return "generate_rag_answer"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count < max_retries:
        return "rewrite_query"

    return "web_search"


# ==========================================
# ROUTE 5: AFTER QUERY REWRITING
# ==========================================

def route_after_query_rewrite(state: dict):

    if state.get("error"):
        return END

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count < max_retries:
        return "retrieve_documents"

    return "web_search"


# ==========================================
# ROUTE 6: AFTER ANSWER GENERATION
# ==========================================

def route_after_generation(state: dict):

    if state.get("error"):
        return END

    return "evaluate_answer"


# ==========================================
# ROUTE 7: SELF-RAG EVALUATION
# ==========================================

def route_after_evaluation(state: dict):

    if state.get("error"):
        return END

    answer_relevant = state.get(
        "answer_relevant", False
    )

    answer_grounded = state.get(
        "answer_grounded", False
    )

    if answer_relevant and answer_grounded:
        return "response"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count < max_retries:
        return "revise_answer"

    return "response"


# ==========================================
# ROUTE 8: AFTER ANSWER REVISION
# ==========================================

def route_after_revision(state: dict):

    if state.get("error"):
        return END

    return "evaluate_answer"


# ==========================================
# ROUTE 9: AFTER WEB SEARCH
# ==========================================

def route_after_web_search(state: dict):

    if state.get("error"):
        return END

    if not state.get("web_results"):
        return "generate_web_answer"

    return "generate_web_answer"


# ==========================================
# ROUTE 10: AFTER FINAL RESPONSE
# ==========================================

def route_after_response(state: dict):

    return END
