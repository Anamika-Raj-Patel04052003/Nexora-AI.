"""
NEXORA AI - LangGraph Conditional Edges

Controls routing between:
- input -> intent
- intent -> chat OR retrieve_documents OR generate_image OR generate_audio OR generate_video OR generate_blog
- retrieve_documents -> grade_documents OR web_search
- grade_documents -> generate_answer OR rewrite_query OR web_search
- rewrite_query -> retrieve_documents OR web_search
- web_search -> generate_answer
- generate_answer -> evaluate_answer
- evaluate_answer -> response OR revise_answer
- revise_answer -> evaluate_answer
- chat / response / creative nodes -> END
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
# ROUTE 2: AFTER INTENT CLASSIFICATION (SUPERVISOR ROUTING)
# ==========================================

def route_by_intent(state: dict):
    if state.get("error"):
        return END

    intent = state.get("intent", "chat")

    if intent == "rag":
        return "retrieve_documents"
    elif intent == "generate_image":
        return "generate_image"
    elif intent == "generate_audio":
        return "generate_audio"
    elif intent == "generate_video":
        return "generate_video"
    elif intent == "generate_blog":
        return "generate_blog"

    return "chat"


# ==========================================
# ROUTE 3: AFTER DOCUMENT RETRIEVAL
# ==========================================

def route_after_retrieval(state: dict):
    if state.get("error"):
        return END

    if state.get("needs_web_search") or not state.get("retrieved_documents"):
        return "web_search"

    return "grade_documents"


# ==========================================
# ROUTE 4: CRAG DOCUMENT GRADING
# ==========================================

def route_after_document_grading(state: dict):
    if state.get("error"):
        return END

    relevant_documents = state.get("relevant_documents", [])
    if relevant_documents:
        return "generate_answer"

    # No relevant chunks found in documents: Check retry budget
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
# ROUTE 6: AFTER WEB SEARCH
# ==========================================

def route_after_web_search(state: dict):
    if state.get("error"):
        return END

    return "generate_answer"


# ==========================================
# ROUTE 7: AFTER ANSWER GENERATION
# ==========================================

def route_after_generation(state: dict):
    if state.get("error"):
        return END

    return "evaluate_answer"


# ==========================================
# ROUTE 8: SELF-RAG EVALUATION
# ==========================================

def route_after_evaluation(state: dict):
    if state.get("error"):
        return END

    answer_relevant = state.get("answer_relevant", True)
    answer_grounded = state.get("answer_grounded", True)

    if answer_relevant and answer_grounded:
        return "response"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count < max_retries:
        return "revise_answer"

    return "response"


# ==========================================
# ROUTE 9: AFTER ANSWER REVISION
# ==========================================

def route_after_revision(state: dict):
    if state.get("error"):
        return END

    return "evaluate_answer"


# ==========================================
# ROUTE 10: AFTER FINAL RESPONSE
# ==========================================

def route_after_response(state: dict):
    return END