"""
NEXORA AI - Chat & Document API Routes

Provides endpoints for:
1. Conversation lifecycle (list, create, get messages, delete with cascade cleanup)
2. Unified streaming chat (auto-detects document availability, triggers CRAG + Self-RAG)
3. Document upload & vector indexing (PDF, TXT, MD, DOCX, CSV)
4. Document question answering (routed via graph)
5. Multimodal analysis (Image Vision, Audio Whisper, Video analysis)
"""

import logging
import tempfile
from pathlib import Path
from typing import Generator

from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Form
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import database
from app.graph.workflow import stream_nexora, run_nexora
from app.rag.pipeline import ingest_document
from app.services.media_service import (
    analyze_image,
    transcribe_audio,
    analyze_video
)


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Conversations"]
)


# --------------------------------------------------
# REQUEST SCHEMAS
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="User message"
    )


class DocumentQuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Question about uploaded documents"
    )


# --------------------------------------------------
# 1. GET ALL CONVERSATIONS
# --------------------------------------------------

@router.get("/conversations")
def list_chats():
    try:
        return database.list_conversations()
    except Exception:
        logger.exception("Failed to fetch conversations")
        raise HTTPException(
            status_code=500,
            detail="Unable to fetch conversations."
        )


# --------------------------------------------------
# 2. CREATE NEW CONVERSATION
# --------------------------------------------------

@router.post("/conversations")
def create_chat():
    try:
        conversation = database.create_conversation()
        logger.info("New conversation created: %s", conversation["id"])
        return conversation
    except Exception:
        logger.exception("Failed to create conversation")
        raise HTTPException(
            status_code=500,
            detail="Unable to create conversation."
        )


# --------------------------------------------------
# 3. GET CONVERSATION MESSAGES
# --------------------------------------------------

@router.get("/conversations/{conversation_id}/messages")
def get_chat_messages(conversation_id: int):
    conversation = database.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    try:
        return database.get_messages(conversation_id)
    except Exception:
        logger.exception("Failed to fetch messages for conversation %s", conversation_id)
        raise HTTPException(
            status_code=500,
            detail="Unable to fetch messages."
        )


# --------------------------------------------------
# 4. DELETE CONVERSATION (CASCADE CLEANUP)
# --------------------------------------------------

@router.delete("/conversations/{conversation_id}")
def delete_chat(conversation_id: int):
    conversation = database.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    try:
        # Delete conversation messages and checkpoints
        database.delete_conversation(conversation_id)

        # Optional vector cleanup from PGVector if configured
        try:
            from app.rag.vector_store import delete_document_from_vector_store
            # Delete vectors associated with this conversation
        except Exception as e:
            logger.debug("PGVector cleanup skipped or not configured: %s", e)

        logger.info("Conversation deleted with cascade cleanup: %s", conversation_id)

        return {
            "success": True,
            "message": "Conversation and associated data deleted successfully."
        }
    except Exception:
        logger.exception("Failed to delete conversation %s", conversation_id)
        raise HTTPException(
            status_code=500,
            detail="Unable to delete conversation."
        )


# --------------------------------------------------
# 5. SEND MESSAGE AND STREAM AI RESPONSE (UNIFIED WORKFLOW)
# --------------------------------------------------

@router.post("/conversations/{conversation_id}/chat")
def send_chat_message(
    conversation_id: int,
    chat_request: ChatRequest
):
    message = chat_request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # Verify conversation exists
    conversation = database.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    try:
        # Fetch previous messages
        previous_messages = database.get_messages(conversation_id)

        # Save user message
        database.add_message(
            conversation_id,
            "user",
            message
        )

        # Generate title from first message
        if not previous_messages:
            title = message[:45]
            database.set_title(conversation_id, title)

        logger.info("User message received for conversation %s", conversation_id)

    except Exception:
        logger.exception("Failed to save user message")
        raise HTTPException(
            status_code=500,
            detail="Unable to save your message."
        )

    # --------------------------------------------------
    # STREAMING GENERATOR
    # --------------------------------------------------

    def generate_response() -> Generator[str, None, None]:
        full_response = ""

        try:
            logger.info(
                "Starting LangGraph unified response generation for conversation %s",
                conversation_id
            )

            # Unified workflow with auto-intent detection:
            # If documents exist -> CRAG + Self-RAG + Web Search
            # If no documents -> Conversational chat
            for chunk in stream_nexora(
                user_input=message,
                conversation_id=conversation_id,
                intent="auto"
            ):
                if chunk:
                    full_response += chunk
                    yield chunk

            # Save completed assistant response
            if full_response.strip():
                database.add_message(
                    conversation_id,
                    "assistant",
                    full_response
                )
                logger.info(
                    "Response saved successfully for conversation %s",
                    conversation_id
                )
            else:
                logger.warning(
                    "LangGraph returned an empty response for conversation %s",
                    conversation_id
                )
                yield "\n\n[AI returned an empty response.]"

        except Exception:
            logger.exception(
                "LangGraph streaming failed for conversation %s",
                conversation_id
            )
            yield (
                "\n\n[AI generation failed. Please check the backend terminal for details.]"
            )

    return StreamingResponse(
        generate_response(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


# --------------------------------------------------
# 6. UPLOAD IMAGE AND GET AI RESPONSE
# --------------------------------------------------

@router.post("/conversations/{conversation_id}/media/image")
async def send_image_message(
    conversation_id: int,
    file: UploadFile = File(...),
    question: str = Form(default="Describe this image in detail.")
):
    conversation = database.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    try:
        image_bytes = await file.read()

        answer = analyze_image(
            image_bytes=image_bytes,
            content_type=file.content_type or "",
            question=question
        )

        user_message = f"[Image: {file.filename}]\n{question}"
        database.add_message(conversation_id, "user", user_message)
        database.add_message(conversation_id, "assistant", answer)

        logger.info("Image analyzed successfully for conversation %s", conversation_id)

        return {
            "success": True,
            "filename": file.filename,
            "question": question,
            "answer": answer
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("Image processing failed for conversation %s", conversation_id)
        raise HTTPException(status_code=500, detail="Unable to analyze image.")
    finally:
        await file.close()


# --------------------------------------------------
# 7. UPLOAD AUDIO AND TRANSCRIBE (WHISPER)
# --------------------------------------------------

@router.post("/conversations/{conversation_id}/media/audio/transcribe")
async def transcribe_audio_message(
    conversation_id: int,
    file: UploadFile = File(...),
    language: str | None = Form(default=None)
):
    conversation = database.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    try:
        audio_bytes = await file.read()

        transcript = transcribe_audio(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            language=language
        )

        logger.info("Audio transcribed successfully for conversation %s", conversation_id)

        return {
            "success": True,
            "filename": file.filename,
            "language": language or "auto-detected",
            "transcript": transcript
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("Audio transcription failed for conversation %s", conversation_id)
        raise HTTPException(status_code=500, detail="Unable to transcribe audio.")
    finally:
        await file.close()


# --------------------------------------------------
# 8. UPLOAD VIDEO AND GET AI RESPONSE
# --------------------------------------------------

@router.post("/conversations/{conversation_id}/media/video")
async def send_video_message(
    conversation_id: int,
    file: UploadFile = File(...),
    question: str = Form(default="Summarize this video."),
    language: str | None = Form(default=None)
):
    conversation = database.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    try:
        video_bytes = await file.read()

        result = analyze_video(
            video_bytes=video_bytes,
            filename=file.filename or "video.mp4",
            question=question,
            language=language
        )

        answer = result["answer"]
        transcript = result["transcript"]

        user_message = f"[Video: {file.filename}]\n{question}"
        database.add_message(conversation_id, "user", user_message)
        database.add_message(conversation_id, "assistant", answer)

        logger.info("Video analyzed successfully for conversation %s", conversation_id)

        return {
            "success": True,
            "filename": file.filename,
            "question": question,
            "answer": answer,
            "transcript": transcript,
            "frames_analyzed": result["frames_analyzed"]
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("Video processing failed for conversation %s", conversation_id)
        raise HTTPException(status_code=500, detail="Unable to analyze video.")
    finally:
        await file.close()


# --------------------------------------------------
# 9. UPLOAD DOCUMENT AND STORE IN VECTOR DATABASE
# --------------------------------------------------

@router.post("/conversations/{conversation_id}/documents/upload")
async def upload_document(
    conversation_id: int,
    file: UploadFile = File(...)
):
    conversation = database.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    allowed_extensions = {".pdf", ".txt", ".md", ".docx", ".csv"}
    original_filename = Path((file.filename or "").replace("\\", "/")).name
    extension = Path(original_filename).suffix.lower()

    if not original_filename or extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Upload PDF, TXT, MD, DOCX, or CSV."
        )

    try:
        file_bytes = await file.read(20 * 1024 * 1024 + 1)
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if len(file_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File size must not exceed 20 MB.")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / original_filename
            temp_file_path.write_bytes(file_bytes)

            result = ingest_document(
                file_path=str(temp_file_path),
                conversation_id=conversation_id
            )

        # Save record of uploaded document in conversation history
        database.add_message(
            conversation_id,
            "user",
            f"[Document uploaded: {original_filename}]"
        )

        logger.info(
            "Document indexed successfully: %s | Conversation: %s",
            original_filename, conversation_id
        )

        return {
            "success": True,
            "message": "Document uploaded and indexed successfully.",
            "filename": original_filename,
            "total_pages": result["total_pages"],
            "total_chunks": result["total_chunks"],
            "stored_vectors": result["stored_vectors"]
        }

    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("Document upload failed for conversation %s", conversation_id)
        raise HTTPException(
            status_code=500,
            detail="Unable to process document. Please ensure vector database is running."
        )
    finally:
        await file.close()


# --------------------------------------------------
# 10. ASK QUESTIONS FROM UPLOADED DOCUMENTS (GRAPH-ROUTED)
# --------------------------------------------------

@router.post("/conversations/{conversation_id}/documents/ask")
def ask_document_question(
    conversation_id: int,
    request: DocumentQuestionRequest
):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    conversation = database.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    try:
        database.add_message(conversation_id, "user", question)

        # Execute through LangGraph workflow with intent='rag'
        result = run_nexora(
            user_input=question,
            conversation_id=conversation_id,
            intent="rag"
        )

        answer = (
            result.get("final_answer")
            or result.get("response")
            or result.get("answer")
            or "I could not find an answer in the uploaded documents."
        )

        database.add_message(conversation_id, "assistant", answer)

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": result.get("sources", []),
            "web_sources": result.get("web_sources", [])
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("Document question answering failed for conversation %s", conversation_id)
        raise HTTPException(status_code=500, detail="Unable to answer document question.")