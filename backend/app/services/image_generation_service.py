
"""
NEXORA AI - Image Generation Service

Cloud-based text-to-image generation using Hugging Face.
"""

import logging
import os
import uuid
from pathlib import Path

from huggingface_hub import InferenceClient


logger = logging.getLogger(__name__)

# Backend directory
BACKEND_DIR = Path(__file__).resolve().parents[2]

# Generated image storage
IMAGE_DIR = BACKEND_DIR / "generated" / "images"

# Image configuration
DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
IMAGE_TIMEOUT = 120
MAX_PROMPT_LENGTH = 1000


def generate_cloud_image(prompt: str) -> dict:
    """
    Generate an image using Hugging Face Inference Providers
    and save it inside the backend generated/images folder.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Image prompt cannot be empty.")

    prompt = prompt.strip()

    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Image prompt cannot exceed {MAX_PROMPT_LENGTH} characters."
        )

    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Configure it in backend/.env."
        )

    model = os.getenv(
        "HF_IMAGE_MODEL",
        DEFAULT_IMAGE_MODEL
    )

    filename = f"{uuid.uuid4().hex}.png"
    image_path = IMAGE_DIR / filename

    try:
        # Initialize Hugging Face client
        client = InferenceClient(
            model=model,
            provider="auto",
            token=token,
            timeout=IMAGE_TIMEOUT
        )

        # Generate image
        generated_image = client.text_to_image(
            prompt=prompt,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT
        )

        if generated_image is None:
            raise RuntimeError(
                "Hugging Face returned an empty image."
            )

        # Save generated image
        IMAGE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        generated_image.save(
            image_path,
            format="PNG"
        )

        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise RuntimeError(
                "Generated image could not be saved correctly."
            )

        logger.info(
            "Image generated successfully using model: %s",
            model
        )

        return {
            "filename": filename,
            "image_path": str(image_path),
            "prompt": prompt,
            "provider": "huggingface",
            "model": model,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "status": "success"
        }

    except Exception as error:

        # Remove incomplete output if generation or saving fails
        if image_path.exists():
            image_path.unlink()

        logger.exception(
            "Cloud image generation failed."
        )

        raise RuntimeError(
            f"Cloud image generation failed: {error}"
        ) from error
