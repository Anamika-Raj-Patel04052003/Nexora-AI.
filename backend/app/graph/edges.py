from langgraph.graph import END


# ==========================================
# ROUTE 1: CHECK INPUT
# ==========================================

def route_after_input(state):

    if state.get("error"):
        return END

    return "intent"


# ==========================================
# ROUTE 2: CHECK INTENT
# ==========================================

def route_by_intent(state):

    intent = state.get("intent", "chat")

    if intent == "rag":
        return "rag"

    return "chat"


# ==========================================
# ROUTE 3: AFTER RESPONSE
# ==========================================

def route_after_response(state):

    if state.get("error"):
        return END

    return END