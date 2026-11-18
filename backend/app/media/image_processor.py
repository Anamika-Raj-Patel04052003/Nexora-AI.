"""
NEXORA AI - Image Processor

Low-level image validation, optimization, and conversion utilities:
- Format & size validation (JPEG, PNG, WEBP)
- Image corruption detection using Pillow
- High-resolution downscaling for Vision API optimization
- Base64 data URI encoding
"""

import base64
import logging
from io import BytesIO
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_DIMENSION = 1024                     # Optimal dimension for Vision models

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
}


def validate_image_bytes(image_bytes: bytes, content_type: str = "") -> bool:
    """
    Ensures image bytes are non-empty, within size limits, and uncorrupted.
    """
    if not image_bytes:
        raise ValueError("Image bytes are empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image size ({len(image_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed 10 MB."
        )

    if content_type and content_type.lower() not in SUPPORTED_MIME_TYPES:
        raise ValueError(f"Unsupported image MIME type: {content_type}. Use JPG, PNG or WEBP.")

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Uploaded file is corrupted or not a valid image format.") from e


def optimize_image_for_vision(
    image_bytes: bytes,
    max_dimension: int = MAX_DIMENSION
) -> tuple[bytes, str]:
    """
    Resizes large images so max dimension does not exceed max_dimension,
    preserving aspect ratio and converting to clean JPEG bytes.
    Returns: (optimized_bytes, mime_type)
    """
    validate_image_bytes(image_bytes)

    with Image.open(BytesIO(image_bytes)) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        width, height = img.size
        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info("Resized image from %dx%d to %dx%d for Vision API.", width, height, *new_size)

        output_buffer = BytesIO()
        img.save(output_buffer, format="JPEG", quality=85, optimize=True)
        return output_buffer.getvalue(), "image/jpeg"


def image_bytes_to_base64_data_url(image_bytes: bytes, content_type: str = "image/jpeg") -> str:
    """
    Converts image bytes to base64 data URI format expected by Groq Vision.
    """
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"