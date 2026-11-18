
"""
NEXORA AI - LLM Service

Centralized Groq LLM initialization and
conversation response utilities.
"""

import logging
from collections.abc import Iterable

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from app.config.settings import settings


logger = logging.getLogger(__name__)


# ----------------------------------------
# INITIALIZE GROQ MODEL
# ----------------------------------------

llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.7,
    streaming=True,
    max_retries=2,
    timeout=60,
)


# ----------------------------------------
# CONVERT DATABASE HISTORY
# INTO LANGCHAIN MESSAGES
# ----------------------------------------

def convert_history(previous_messages):

    if previous_messages is None:
        return []

    if not isinstance(previous_messages, Iterable):
        raise ValueError("Conversation history must be iterable.")

    history = []

    for item in previous_messages:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if not isinstance(content, str) or not content.strip():
            continue

        if role == "user":

            history.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":

            history.append(
                AIMessage(content=content)
            )

    return history


# ----------------------------------------
# NORMAL CHAT RESPONSE
# ----------------------------------------

def ask(message: str) -> str:

    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message cannot be empty.")

    from app.chains.chat_chain import generate_chat_response

    try:

        return generate_chat_response(
            message.strip(),
            []
        )

    except Exception as error:

        logger.exception("Normal chat response generation failed.")

        raise RuntimeError(
            "Unable to generate chat response."
        ) from error


# ----------------------------------------
# STREAMING CHAT RESPONSE
# ----------------------------------------

def stream_reply(previous_messages, message: str):

    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message cannot be empty.")

    from app.chains.chat_chain import stream_chat_response

    history = convert_history(previous_messages)

    try:

        yield from stream_chat_response(
            message.strip(),
            history
        )

    except Exception:

        logger.exception("Streaming chat response failed.")

        raise
