
"""
NEXORA AI - Audio Generation Service

Text-to-Speech generation using Edge TTS.
Generates standalone MP3 audio files.
"""

import logging
import uuid

import edge_tts

from app.services.audio_generation_config import (
    AUDIO_DIR,
    DEFAULT_VOICE,
    MAX_TEXT_LENGTH,
    MAX_VOICE_LENGTH
)


logger = logging.getLogger(__name__)


async def generate_audio(
    text: str,
    voice: str = DEFAULT_VOICE
) -> dict:
    """
    Convert text into speech using Edge TTS.
    Returns generated audio metadata.
    """

    # ----------------------------------------------
    # INPUT VALIDATION
    # ----------------------------------------------

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text cannot be empty.")

    text = text.strip()

    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Text must not exceed {MAX_TEXT_LENGTH} characters."
        )

    if not isinstance(voice, str) or not voice.strip():
        raise ValueError("Invalid voice selection.")

    voice = voice.strip()

    if len(voice) > MAX_VOICE_LENGTH:
        raise ValueError("Voice selection is too long.")

    # ----------------------------------------------
    # FILE CONFIGURATION
    # ----------------------------------------------

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.mp3"
    audio_path = AUDIO_DIR / filename

    success = False

    try:

        # ------------------------------------------
        # EDGE TTS GENERATION
        # ------------------------------------------

        communicator = edge_tts.Communicate(
            text,
            voice
        )

        await communicator.save(str(audio_path))

        # ------------------------------------------
        # OUTPUT VALIDATION
        # ------------------------------------------

        if (
            not audio_path.is_file()
            or audio_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Generated audio file is empty."
            )

        success = True

        logger.info(
            "Audio generated successfully. Voice: %s",
            voice
        )

        return {
            "filename": filename,
            "audio_path": str(audio_path),
            "voice": voice,
            "provider": "edge-tts",
            "character_count": len(text),
            "status": "success"
        }

    except ValueError:
        raise

    except Exception as error:

        logger.exception("Audio generation failed.")

        raise RuntimeError(
            "Audio generation failed. Please try again."
        ) from error

    finally:

        # ------------------------------------------
        # CLEANUP FAILED GENERATION
        # ------------------------------------------

        if not success and audio_path.exists():

            try:
                audio_path.unlink()

            except OSError:

                logger.warning(
                    "Could not remove incomplete audio file: %s",
                    audio_path
                )
