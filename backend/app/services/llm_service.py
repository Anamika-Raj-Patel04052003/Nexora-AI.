
from langchain_groq import ChatGroq

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from app.config.settings import settings


# ----------------------------------------
# INITIALIZE GROQ MODEL
# ----------------------------------------

llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.7,
    streaming=True,
    max_retries=2,
    timeout=60
)


# ----------------------------------------
# CONVERT DATABASE HISTORY
# INTO LANGCHAIN MESSAGES
# ----------------------------------------

def convert_history(previous_messages):

    history = []

    for item in previous_messages:

        role = item["role"]
        content = item["content"]

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

    from app.chains.chat_chain import (
        generate_chat_response
    )

    return generate_chat_response(
        message,
        []
    )


# ----------------------------------------
# STREAMING CHAT RESPONSE
# ----------------------------------------

def stream_reply(previous_messages, message):

    from app.chains.chat_chain import (
        stream_chat_response
    )

    history = convert_history(
        previous_messages
    )

    yield from stream_chat_response(
        message,
        history
    )
