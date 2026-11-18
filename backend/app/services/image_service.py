
"""
NEXORA AI - Unified Image Generation Service

Supports:
1. Cloud image generation using Hugging Face.
2. Local image generation using Diffusers.
"""

import logging

from app.services.image_generation_service import generate_cloud_image
from app.services.local_image_generation_service import generate_local_image


logger = logging.getLogger(__name__)

SUPPORTED_MODES = {"cloud", "local"}
MAX_PROMPT_LENGTH = 1000


def generate_image(
    prompt: str,
    mode: str = "cloud"
) -> dict:
    """
    Select the image generation provider based on mode.
    """

    # Validate prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Image prompt cannot be empty.")

    prompt = prompt.strip()

    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Image prompt cannot exceed {MAX_PROMPT_LENGTH} characters."
        )

    # Validate mode
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(
            "Image generation mode cannot be empty."
        )

    mode = mode.strip().lower()

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            "Invalid image generation mode. Choose 'cloud' or 'local'."
        )

    try:
        # Select image generation provider
        if mode == "cloud":
            result = generate_cloud_image(prompt)

        else:
            result = generate_local_image(prompt)

        logger.info(
            "Image generated successfully using %s mode.",
            mode
        )

        return {
            "status": "success",
            "mode": mode,
            "filename": result["filename"],
            "image_path": result["image_path"],
            "prompt": result["prompt"],
            "provider": result["provider"],
            "model": result["model"],
            "width": result.get("width"),
            "height": result.get("height")
        }

    except ValueError:
        raise

    except Exception as error:
        logger.exception(
            "Image generation failed in %s mode.",
            mode
        )

        raise RuntimeError(
            f"Image generation failed in {mode} mode: {error}"
        ) from error
