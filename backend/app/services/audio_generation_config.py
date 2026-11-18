
"""
NEXORA AI - Shared Audio Configuration
"""

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]

AUDIO_DIR = BACKEND_DIR / "generated" / "audio"

DEFAULT_VOICE = "en-US-AriaNeural"

MAX_TEXT_LENGTH = 5000

MAX_VOICE_LENGTH = 100

MAX_CHUNK_SIZE = 4500
