"""
NEXORA AI - Short-Term Memory Service

Manages dynamic sliding window history and automatic
conversation summarization for active chat sessions.
Prevents context window overflow while preserving critical conversational flow.
"""

import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app import database
from app.services.llm_service import llm

logger = logging.getLogger(__name__)

DEFAULT_SLIDING_WINDOW = 8  # Keeps the last 8 messages verbatim
SUMMARY_THRESHOLD = 12       # If more than 12 messages, summarizes the older ones


def get_conversation_history(conversation_id: int) -> list[dict]:
    """
    Fetch raw message history from SQLite for a conversation.
    """
    try:
        return database.get_messages(conversation_id)
    except Exception as e:
        logger.warning("Could not fetch messages for conversation %s: %s", conversation_id, e)
        return []


def summarize_messages(messages_to_summarize: list[dict]) -> str:
    """
    Condenses older conversation turns into a high-density factual summary.
    """
    if not messages_to_summarize:
        return ""

    formatted_transcript = []
    for msg in messages_to_summarize:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        formatted_transcript.append(f"{role}: {content}")

    transcript_text = "\n".join(formatted_transcript)

    summary_prompt = f"""
You are an expert conversation summarizer for NEXORA AI.
Summarize the key topics, user questions, decisions, and facts discussed in the following conversation history.
Keep the summary concise (under 150 words), clear, and strictly factual.

CONVERSATION TRANSCRIPT:
{transcript_text}

CONCISE FACTUAL SUMMARY:
"""

    try:
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content.strip()
        logger.info("Successfully compressed %d older messages into short-term summary.", len(messages_to_summarize))
        return summary
    except Exception as e:
        logger.warning("Summarization fallback: %s", e)
        return "Previous discussion covered general queries and responses."


def get_optimized_chat_context(
    conversation_id: int,
    sliding_window: int = DEFAULT_SLIDING_WINDOW
) -> list:
    """
    Returns an optimized list of LangChain messages:
    - If total messages <= SUMMARY_THRESHOLD: Returns all messages as Human/AIMessage.
    - If total messages > SUMMARY_THRESHOLD:
      - Summarizes all messages before the sliding window.
      - Injects the summary as a SystemMessage.
      - Follows with the recent sliding_window messages verbatim.
    """
    raw_messages = get_conversation_history(conversation_id)
    if not raw_messages:
        return []

    if len(raw_messages) <= SUMMARY_THRESHOLD:
        formatted = []
        for item in raw_messages:
            role = item.get("role")
            content = item.get("content", "")
            if not content.strip():
                continue
            if role == "user":
                formatted.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted.append(AIMessage(content=content))
        return formatted

    # When history is long: Split into older (to summarize) and recent (verbatim)
    older_messages = raw_messages[:-sliding_window]
    recent_messages = raw_messages[-sliding_window:]

    summary_text = summarize_messages(older_messages)

    optimized_messages = []

    if summary_text:
        context_system_message = (
            f"[PREVIOUS CONVERSATION CONTEXT SUMMARY]\n"
            f"{summary_text}\n"
            f"[END OF CONTEXT SUMMARY]"
        )
        optimized_messages.append(SystemMessage(content=context_system_message))

    for item in recent_messages:
        role = item.get("role")
        content = item.get("content", "")
        if not content.strip():
            continue
        if role == "user":
            optimized_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            optimized_messages.append(AIMessage(content=content))

    return optimized_messages