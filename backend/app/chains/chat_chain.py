
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.output_parsers import StrOutputParser

from app.services.llm_service import llm


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

You can use retrieved document context when it is
provided by the RAG workflow.

If document context is provided, prioritize that
context for document-specific questions.
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
# INVOKE CHAIN
# ----------------------------------------

def generate_chat_response(
    message: str,
    history: list
) -> str:

    response = chat_chain.invoke(
        {
            "input": message,
            "history": history
        }
    )

    return response


# ----------------------------------------
# STREAM CHAIN
# ----------------------------------------

def stream_chat_response(
    message: str,
    history: list
):

    for chunk in chat_chain.stream(
        {
            "input": message,
            "history": history
        }
    ):

        if chunk:
            yield chunk
