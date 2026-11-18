"""
NEXORA AI - Supervisor Agent
"""

import json
import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage

from app.services.llm_service import llm

# Safe Fallback Import
try:
    from app.database import has_uploaded_documents
except ImportError:
    def has_uploaded_documents(conversation_id: int) -> bool:
        try:
            from app.database import get_messages
            msgs = get_messages(conversation_id)
            return any(m.get("content", "").startswith("[Document uploaded:") for m in msgs)
        except Exception:
            return False

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """
You are the central Supervisor Router of NEXORA AI.
Classify the user's intent into EXACTLY ONE of the following categories:

1. "generate_image" -> User wants to generate, draw, create, or render an image or picture.
2. "generate_audio" -> User explicitly wants speech, voice, TTS, audio recitation.
3. "generate_video" -> User wants to compile or create an MP4 video slideshow.
4. "generate_blog" -> User asks to write a blog, article, or in-depth guide with sections.
5. "rag" -> User asks a question about an uploaded document, PDF, file, or data.
6. "chat" -> General conversational questions, programming help, creative chat, or reasoning.

Respond with ONLY valid JSON:
{"intent": "generate_image"|"generate_audio"|"generate_video"|"generate_blog"|"rag"|"chat", "cleaned_prompt": "extracted core prompt for generation"}
"""


def fast_keyword_intent_check(user_input: str) -> str | None:
    text = user_input.lower().strip()

    if re.search(r"\b(generate|create|draw|render|make)\s+(an?\s+)?(image|photo|picture|illustration|art)\b", text) or \
       re.search(r"\b(image|photo|picture)\s+(banao|generate|create)\b", text):
        return "generate_image"

    if re.search(r"\b(generate|create|convert\s+to|make)\s+(audio|speech|voice|podcast|tts)\b", text) or \
       re.search(r"\b(audio|voice|awaz)\s+(sunao|banao|nikalo)\b", text):
        return "generate_audio"

    if re.search(r"\b(generate|create|make)\s+(an?\s+)?(video|slideshow)\b", text) or \
       re.search(r"\b(video)\s+(banao|generate)\b", text):
        return "generate_video"

    if re.search(r"\b(write|create|generate)\s+(an?\s+)?(blog|article|publication)\b", text) or \
       re.search(r"\b(blog|article)\s+(likho|banao)\b", text):
        return "generate_blog"

    return None


def classify_user_intent(
    user_input: str,
    conversation_id: int | str | None = None
) -> dict:
    clean_input = user_input.strip()

    fast_intent = fast_keyword_intent_check(clean_input)
    if fast_intent:
        logger.info("Fast keyword detector matched intent: %s", fast_intent)
        return {"intent": fast_intent, "cleaned_prompt": clean_input}

    has_docs = False
    if conversation_id is not None:
        try:
            has_docs = has_uploaded_documents(int(conversation_id))
        except Exception:
            pass

    try:
        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Has uploaded documents: {has_docs}\nUser message: {clean_input}")
        ]

        response = llm.invoke(messages)
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(content)

        intent = data.get("intent", "chat")
        cleaned_prompt = data.get("cleaned_prompt") or clean_input

        if intent == "chat" and has_docs:
            intent = "rag"

        logger.info("Supervisor Agent routed query to: %s", intent)
        return {"intent": intent, "cleaned_prompt": cleaned_prompt}

    except Exception as e:
        logger.warning("Supervisor classification fallback to auto: %s", e)
        if has_docs:
            return {"intent": "rag", "cleaned_prompt": clean_input}
        return {"intent": "chat", "cleaned_prompt": clean_input}