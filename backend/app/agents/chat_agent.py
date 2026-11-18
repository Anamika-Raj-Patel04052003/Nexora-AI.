"""
NEXORA AI - Chat Agent

Dedicated conversational AI agent integrating:
- Short-term sliding window context (short_term.py)
- Long-term user memories and preferences (long_term.py)
- Groq LLM streaming and synchronous generation
"""

import logging
from app.chains.chat_chain import generate_chat_response, stream_chat_response
from app.memory.short_term import get_optimized_chat_context
from app.memory.long_term import extract_and_learn_facts


logger = logging.getLogger(__name__)


class ChatAgent:
    """
    Manages conversational responses with short-term context compression
    and cross-session memory integration.
    """

    def __init__(self):
        self.agent_name = "NEXORA Chat Agent"

    def respond(
        self,
        message: str,
        conversation_id: int | None = None
    ) -> str:
        clean_message = message.strip()

        # Learn user facts in background
        try:
            extract_and_learn_facts(clean_message)
        except Exception as e:
            logger.debug("Fact extraction skipped: %s", e)

        # Get optimized sliding window history with summarization
        history = (
            get_optimized_chat_context(conversation_id)
            if conversation_id is not None
            else []
        )

        return generate_chat_response(
            message=clean_message,
            history=history
        )

    def stream_respond(
        self,
        message: str,
        conversation_id: int | None = None
    ):
        clean_message = message.strip()

        # Learn user facts in background
        try:
            extract_and_learn_facts(clean_message)
        except Exception as e:
            logger.debug("Fact extraction skipped: %s", e)

        history = (
            get_optimized_chat_context(conversation_id)
            if conversation_id is not None
            else []
        )

        yield from stream_chat_response(
            message=clean_message,
            history=history
        )


chat_agent = ChatAgent()