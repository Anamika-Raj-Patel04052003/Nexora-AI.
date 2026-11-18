"""
NEXORA AI - Audio Processor

Low-level audio verification, format validation,
and temporary storage management:
- Extension and size checks (MP3, WAV, M4A, OGG, FLAC, WEBM)
- Whisper API input preparation
- Audio chunking and sanitization utilities
"""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB (Groq Whisper limit)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".webm"
}


def validate_audio_file(audio_bytes: bytes, filename: str) -> bool:
    """
    Validates that audio bytes are present, within limits,
    and possess a supported audio extension.
    """
    if not audio_bytes:
        raise ValueError("Audio data is empty.")

    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise ValueError(
            f"Audio file size ({len(audio_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed 25 MB."
        )

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{extension}'. Supported: MP3, WAV, M4A, OGG, FLAC, WEBM."
        )

    return True


def save_audio_temporarily(audio_bytes: bytes, extension: str = ".wav") -> str:
    """
    Saves audio bytes to a temporary file on disk and returns its path.
    Caller is responsible for cleaning up the temporary file when finished.
    """
    if not extension.startswith("."):
        extension = f".{extension}"

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
        temp_file.write(audio_bytes)
        return temp_file.name


def safe_cleanup_file(file_path: str | Path | None) -> None:
    """
    Silently removes a file if it exists.
    """
    if not file_path:
        return
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
    except OSError as e:
        logger.debug("Could not remove temp file %s: %s", file_path, e)