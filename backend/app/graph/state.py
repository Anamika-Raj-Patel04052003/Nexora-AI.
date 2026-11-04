
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage

from langgraph.graph.message import add_messages


class NexoraState(TypedDict, total=False):

    # --------------------------------
    # CONVERSATION MESSAGES
    # --------------------------------

    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]

    # --------------------------------
    # CURRENT USER INPUT
    # --------------------------------

    user_input: str

    # --------------------------------
    # DETECTED USER INTENT
    # --------------------------------

    intent: str

    # --------------------------------
    # GENERATED RESPONSE
    # --------------------------------

    response: str

    # --------------------------------
    # CONVERSATION ID
    # --------------------------------

    conversation_id: int

    # --------------------------------
    # RETRIEVED DOCUMENT CONTEXT
    # --------------------------------

    retrieved_context: str

    # --------------------------------
    # RETRIEVED DOCUMENT SOURCES
    # --------------------------------

    sources: list[dict[str, str | int]]

    # --------------------------------
    # UPLOADED FILE INFORMATION
    # --------------------------------

    file_path: str

    # --------------------------------
    # ERROR INFORMATION
    # --------------------------------

    error: str
