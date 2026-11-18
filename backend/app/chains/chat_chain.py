"""
NEXORA AI - Chat Chain with Long-Term Memory

Handles conversational responses with:
- Dynamic chat history
- Long-term user profile & preferences injection (long_term.py)
- Automatic background learning of user facts
"""

import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.services.llm_service import llm
from app.memory.long_term import format_long_term_memories, extract_and_learn_facts


logger = logging.getLogger(__name__)


# ----------------------------------------
# BASE SYSTEM PROMPT
# ----------------------------------------

BASE_SYSTEM_PROMPT = """
You are NEXORA AI, an advanced multilingual AI assistant.

Your responsibilities:
1. Understand the user's question carefully and respond naturally.
2. Reply in the same language as the user's question (Hindi, English, Hinglish, etc.).
3. Maintain conversation context and personalize responses using the USER PROFILE & MEMORIES provided below.
4. Give accurate, clear, and well-structured answers using Markdown.
5. Explain programming and technical concepts with clear, concise examples.
6. Be friendly, highly capable, professional, and helpful.

{memory_context}
"""


# ----------------------------------------
# CHAT PROMPT TEMPLATE
# ----------------------------------------

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", BASE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)


# ----------------------------------------
# LANGCHAIN LCEL CHAIN
# ----------------------------------------

chat_chain = (
    chat_prompt
    | llm
    | StrOutputParser()
)


# ----------------------------------------
# VALIDATE INPUT
# ----------------------------------------

def validate_chat_input(message: str, history: list):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message cannot be empty.")
    if not isinstance(history, list):
        raise ValueError("Chat history must be a list.")


# ----------------------------------------
# INVOKE CHAIN (WITH MEMORY)
# ----------------------------------------

def generate_chat_response(
    message: str,
    history: list
) -> str:
    validate_chat_input(message, history)
    clean_message = message.strip()

    # Silently extract and remember any personal facts/preferences shared
    try:
        extract_and_learn_facts(clean_message)
    except Exception as e:
        logger.debug("Fact extraction skipped: %s", e)

    # Format long-term memories for injection
    memory_context = format_long_term_memories()

    try:
        response = chat_chain.invoke(
            {
                "input": clean_message,
                "history": history,
                "memory_context": memory_context
            }
        )

        if not response or not response.strip():
            raise RuntimeError("The chat model returned an empty response.")

        return response.strip()

    except Exception as error:
        logger.exception("Chat response generation failed: %s", error)
        raise RuntimeError("Unable to generate chat response.") from error


# ----------------------------------------
# STREAM CHAIN (WITH MEMORY)
# ----------------------------------------

def stream_chat_response(
    message: str,
    history: list
):
    validate_chat_input(message, history)
    clean_message = message.strip()

    # Silently extract and remember any personal facts/preferences shared
    try:
        extract_and_learn_facts(clean_message)
    except Exception as e:
        logger.debug("Fact extraction skipped: %s", e)

    # Format long-term memories for injection
    memory_context = format_long_term_memories()

    try:
        for chunk in chat_chain.stream(
            {
                "input": clean_message,
                "history": history,
                "memory_context": memory_context
            }
        ):
            if chunk:
                yield chunk

    except Exception as error:
        logger.exception("Chat response streaming failed: %s", error)
        raise RuntimeError("Unable to stream chat response.") from error