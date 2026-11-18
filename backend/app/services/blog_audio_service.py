
"""
NEXORA AI - Blog to Audio Service

Converts long Markdown blogs into one MP3 narration
using Edge TTS and MoviePy.
"""

import asyncio
import html
import logging
import re
import uuid

from pathlib import Path

import edge_tts

from moviepy import AudioFileClip, concatenate_audioclips

from app.services.audio_generation_config import (
    AUDIO_DIR,
    DEFAULT_VOICE,
    MAX_CHUNK_SIZE,
    MAX_VOICE_LENGTH
)


logger = logging.getLogger(__name__)


# ==================================================
# 1. MARKDOWN CLEANING
# ==================================================

def clean_blog_text(content: str) -> str:
    """Convert Markdown blog content into readable narration text."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Blog content cannot be empty.")

    text = content

    # Remove NEXORA image placeholders
    text = re.sub(
        r"<!--\s*NEXORA_IMAGE:.*?-->",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove fenced code blocks
    text = re.sub(
        r"```.*?```",
        " ",
        text,
        flags=re.DOTALL
    )

    # Remove Markdown images
    text = re.sub(
        r"!\[[^\]]*\]\([^)]+\)",
        " ",
        text
    )

    # Keep link text and remove URLs
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Decode HTML entities
    text = html.unescape(text)

    # Remove Markdown headings
    text = re.sub(
        r"^#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove bold formatting
    text = re.sub(
        r"(\*\*|__)(.*?)\1",
        r"\2",
        text,
        flags=re.DOTALL
    )

    # Remove italic formatting
    text = re.sub(
        r"(?<!\w)([*_])([^*_]+)\1(?!\w)",
        r"\2",
        text
    )

    # Remove inline code markers
    text = re.sub(
        r"`([^`]+)`",
        r"\1",
        text
    )

    # Remove Markdown table separator rows
    text = re.sub(
        r"^\s*\|?[\s:|-]+\|?\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Convert table separators into narration pauses
    text = text.replace("|", ". ")

    # Remove unordered list markers
    text = re.sub(
        r"^\s*[-*+]\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove numbered list markers
    text = re.sub(
        r"^\s*\d+\.\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # Normalize spaces while preserving paragraphs
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ==================================================
# 2. TEXT CHUNKING
# ==================================================

def split_text(text: str) -> list[str]:
    """
    Split narration text into manageable chunks.

    Preserves paragraph boundaries where possible.
    No chunk exceeds MAX_CHUNK_SIZE characters.
    """

    if not isinstance(MAX_CHUNK_SIZE, int) or MAX_CHUNK_SIZE <= 0:
        raise ValueError("MAX_CHUNK_SIZE must be a positive integer.")

    if not text or not text.strip():
        return []

    chunks = []
    current_chunk = ""

    for paragraph in text.splitlines():

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Split exceptionally long paragraphs
        while len(paragraph) > MAX_CHUNK_SIZE:

            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            split_at = paragraph.rfind(
                " ",
                0,
                MAX_CHUNK_SIZE + 1
            )

            if split_at <= 0:
                split_at = MAX_CHUNK_SIZE

            chunks.append(
                paragraph[:split_at].strip()
            )

            paragraph = paragraph[split_at:].strip()

        candidate = (
            f"{current_chunk}\n\n{paragraph}"
            if current_chunk
            else paragraph
        )

        if len(candidate) <= MAX_CHUNK_SIZE:
            current_chunk = candidate

        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return [chunk for chunk in chunks if chunk]


# ==================================================
# 3. AUDIO MERGING
# ==================================================

def merge_audio_chunks(
    temporary_files: list[Path],
    output_path: Path
) -> None:
    """Merge generated MP3 chunks into one MP3 file."""

    audio_clips = []
    final_clip = None

    try:

        # Load individual audio files
        for temp_path in temporary_files:

            if not temp_path.is_file() or temp_path.stat().st_size == 0:
                raise RuntimeError(
                    "An audio chunk is missing or empty."
                )

            clip = AudioFileClip(str(temp_path))
            audio_clips.append(clip)

        if not audio_clips:
            raise RuntimeError(
                "No audio chunks available for merging."
            )

        # Combine audio clips
        final_clip = concatenate_audioclips(audio_clips)

        # Export final narration
        final_clip.write_audiofile(
            str(output_path),
            fps=24000,
            codec="libmp3lame",
            bitrate="128k",
            logger=None
        )

    finally:

        # Close MoviePy resources
        if final_clip is not None:
            final_clip.close()

        for clip in audio_clips:
            clip.close()


# ==================================================
# 4. BLOG AUDIO GENERATION
# ==================================================

async def generate_blog_audio(
    blog_content: str,
    voice: str = DEFAULT_VOICE
) -> dict:
    """
    Convert an entire Markdown blog into one MP3 file.

    Steps:
    1. Clean Markdown.
    2. Split text into chunks.
    3. Generate audio using Edge TTS.
    4. Merge chunks using MoviePy.
    5. Remove temporary files.
    """

    # ----------------------------------------------
    # VOICE VALIDATION
    # ----------------------------------------------

    if not isinstance(voice, str) or not voice.strip():
        raise ValueError("Invalid voice selection.")

    voice = voice.strip()

    if len(voice) > MAX_VOICE_LENGTH:
        raise ValueError("Voice selection is too long.")

    # ----------------------------------------------
    # CLEAN BLOG CONTENT
    # ----------------------------------------------

    cleaned_text = clean_blog_text(blog_content)

    if not cleaned_text:
        raise ValueError(
            "No readable text found in the blog."
        )

    # ----------------------------------------------
    # SPLIT TEXT
    # ----------------------------------------------

    chunks = split_text(cleaned_text)

    if not chunks:
        raise ValueError(
            "Unable to prepare blog narration."
        )

    # ----------------------------------------------
    # FILE CONFIGURATION
    # ----------------------------------------------

    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = f"blog_{uuid.uuid4().hex}.mp3"
    audio_path = AUDIO_DIR / filename

    temporary_files = []
    success = False

    try:

        # ------------------------------------------
        # GENERATE AUDIO CHUNKS
        # ------------------------------------------

        for index, chunk in enumerate(chunks):

            temp_filename = (
                f"temp_{uuid.uuid4().hex}_{index}.mp3"
            )

            temp_path = AUDIO_DIR / temp_filename

            temporary_files.append(temp_path)

            communicator = edge_tts.Communicate(
                chunk,
                voice
            )

            await communicator.save(str(temp_path))

            if (
                not temp_path.is_file()
                or temp_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    f"Audio generation failed for chunk {index + 1}."
                )

        # ------------------------------------------
        # MERGE AUDIO CHUNKS
        # ------------------------------------------

        await asyncio.to_thread(
            merge_audio_chunks,
            temporary_files,
            audio_path
        )

        # ------------------------------------------
        # VALIDATE FINAL AUDIO
        # ------------------------------------------

        if (
            not audio_path.is_file()
            or audio_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Final narration file is empty."
            )

        success = True

        logger.info(
            "Blog narration generated successfully. Chunks: %s",
            len(chunks)
        )

        return {
            "filename": filename,
            "audio_path": str(audio_path),
            "voice": voice,
            "provider": "edge-tts",
            "chunk_count": len(chunks),
            "character_count": len(cleaned_text),
            "status": "success"
        }

    except ValueError:
        raise

    except Exception as error:

        logger.exception(
            "Blog audio generation failed."
        )

        raise RuntimeError(
            "Blog narration failed. Please try again."
        ) from error

    finally:

        # ------------------------------------------
        # CLEANUP TEMPORARY FILES
        # ------------------------------------------

        for temp_path in temporary_files:

            try:
                if temp_path.exists():
                    temp_path.unlink()

            except OSError:

                logger.warning(
                    "Could not remove temporary audio file: %s",
                    temp_path
                )

        # Remove incomplete final output
        if not success:

            try:
                if audio_path.exists():
                    audio_path.unlink()

            except OSError:

                logger.warning(
                    "Could not remove incomplete audio file: %s",
                    audio_path
                )
