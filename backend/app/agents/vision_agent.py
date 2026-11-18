"""
NEXORA AI - Vision & Multimodal Agent

Dedicated agent for analyzing visual and audio inputs:
- Image understanding & optical reasoning via Groq Vision
- Audio transcription via Groq Whisper
- Video frame extraction + audio transcription combined synthesis
"""

import logging
from app.services.media_service import (
    analyze_image,
    transcribe_audio,
    analyze_video
)

logger = logging.getLogger(__name__)


class VisionAgent:
    """
    Coordinates visual understanding, video frame analysis,
    and speech recognition.
    """

    def __init__(self):
        self.agent_name = "NEXORA Vision Agent"

    def inspect_image(
        self,
        image_bytes: bytes,
        content_type: str,
        question: str = "Describe this image in detail."
    ) -> str:
        """
        Analyze an uploaded image with Groq Vision.
        """
        return analyze_image(
            image_bytes=image_bytes,
            content_type=content_type,
            question=question
        )

    def process_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        language: str | None = None
    ) -> str:
        """
        Transcribe spoken audio using Groq Whisper.
        """
        return transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            language=language
        )

    def inspect_video(
        self,
        video_bytes: bytes,
        filename: str,
        question: str = "Summarize this video.",
        language: str | None = None
    ) -> dict:
        """
        Extract video frames and audio, transcribing dialogue and
        reasoning across visual events.
        """
        return analyze_video(
            video_bytes=video_bytes,
            filename=filename,
            question=question,
            language=language
        )


vision_agent = VisionAgent()