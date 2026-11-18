
"""
NEXORA AI - Chat Chain

Handles normal conversational responses with chat history.
"""

import logging

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.output_parsers import StrOutputParser

from app.services.llm_service import llm


logger = logging.getLogger(__name__)


# ----------------------------------------
# SYSTEM PROMPT
# ----------------------------------------

SYSTEM_PROMPT = """
You are NEXORA AI, an advanced multilingual AI assistant.

Your responsibilities:

1. Understand the user's question carefully.
2. Respond naturally and intelligently.
3. Reply in the same language as the user's question.
4. Support Hindi, English and other languages.
5. Maintain conversation context.
6. Give accurate and well-structured answers.
7. Use Markdown when appropriate.
8. Explain programming concepts with examples.
9. Never invent facts when information is unavailable.
10. Be friendly, professional and helpful.

For normal chat requests, answer using your general
knowledge while being transparent about uncertainty.

Document-specific questions should be handled by
the document RAG workflow.

If document context is provided explicitly, prioritize
that context for document-specific questions.
"""


# ----------------------------------------
# CHAT PROMPT TEMPLATE
# ----------------------------------------

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),

        MessagesPlaceholder(
            variable_name="history"
        ),

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
# INVOKE CHAIN
# ----------------------------------------

def generate_chat_response(
    message: str,
    history: list
) -> str:

    validate_chat_input(message, history)

    try:

        response = chat_chain.invoke(
            {
                "input": message.strip(),
                "history": history
            }
        )

        if not response or not response.strip():
            raise RuntimeError(
                "The chat model returned an empty response."
            )

        return response.strip()

    except Exception as error:

        logger.exception("Chat response generation failed.")

        raise RuntimeError(
            "Unable to generate chat response."
        ) from error


# ----------------------------------------
# STREAM CHAIN
# ----------------------------------------

def stream_chat_response(
    message: str,
    history: list
):

    validate_chat_input(message, history)

    try:

        for chunk in chat_chain.stream(
            {
                "input": message.strip(),
                "history": history
            }
        ):

            if chunk:
                yield chunk

    except Exception as error:

        logger.exception("Chat response streaming failed.")

        raise RuntimeError(
            "Unable to stream chat response."
        ) from error
