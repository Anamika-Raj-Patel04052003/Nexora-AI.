"""
NEXORA AI - Video Processor

Video verification, audio track extraction, and FFmpeg execution utilities:
- Size & extension validation (MP4, WEBM up to 25 MB)
- High-efficiency FFmpeg audio track extraction (PCM 16kHz mono)
- Temporary input/output stream management
- Whisper transcription preparation from video
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
import imageio_ffmpeg

logger = logging.getLogger(__name__)

MAX_VIDEO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm"}


def validate_video_file(video_bytes: bytes, filename: str) -> bool:
    """
    Validates presence, size limits, and format extension for an uploaded video.
    """
    if not video_bytes:
        raise ValueError("Video data is empty.")

    if len(video_bytes) > MAX_VIDEO_SIZE_BYTES:
        raise ValueError(
            f"Video size ({len(video_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed 25 MB."
        )

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format '{ext}'. Upload MP4 or WEBM.")

    return True


def extract_audio_from_video(video_bytes: bytes, filename: str) -> bytes:
    """
    Uses FFmpeg via imageio_ffmpeg to isolate the audio track from a video
    and returns 16kHz mono 16-bit PCM WAV bytes optimal for Whisper transcription.
    """
    validate_video_file(video_bytes, filename)

    input_path = None
    output_path = None

    try:
        suffix = Path(filename).suffix.lower() or ".mp4"

        # Write uploaded video to temporary file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as in_file:
            in_file.write(video_bytes)
            input_path = in_file.name

        # Allocate temporary WAV output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            output_path = out_file.name

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_exe,
            "-y",
            "-i", input_path,
            "-vn",                # Drop visual stream
            "-ac", "1",           # Mono channel
            "-ar", "16000",       # 16kHz sample rate (Whisper standard)
            "-c:a", "pcm_s16le",  # Uncompressed 16-bit PCM
            output_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise ValueError(
                f"FFmpeg audio extraction failed: {result.stderr[:200] if result.stderr else 'No audio track detected.'}"
            )

        with open(output_path, "rb") as af:
            extracted_audio_bytes = af.read()

        if not extracted_audio_bytes:
            raise ValueError("Extracted audio stream is empty.")

        logger.info(
            "Extracted %.2f KB audio track from video '%s'.",
            len(extracted_audio_bytes) / 1024,
            filename
        )

        return extracted_audio_bytes

    finally:
        for p in [input_path, output_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass