
"""
NEXORA AI - Experimental Local Image Generation

Runs a lightweight Diffusers model on CPU.
The model is loaded lazily and reused across requests.
"""

import logging
import os
import threading
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = BACKEND_DIR / "generated" / "images"

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "segmind/tiny-sd"

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
INFERENCE_STEPS = 4
MAX_PROMPT_LENGTH = 1000

# Model is loaded only when local generation is requested.
_pipeline = None

# Prevent multiple simultaneous model initializations.
_model_lock = threading.Lock()

# Prevent concurrent CPU inference requests.
_generation_lock = threading.Lock()


def get_local_pipeline():
    """
    Load the Diffusers pipeline once and reuse it.
    """

    global _pipeline

    if _pipeline is not None:
        return _pipeline

    with _model_lock:

        if _pipeline is not None:
            return _pipeline

        try:
            import torch
            from diffusers import AutoPipelineForText2Image

            model_id = os.getenv(
                "LOCAL_IMAGE_MODEL",
                DEFAULT_LOCAL_MODEL
            )

            logger.info(
                "Loading local image model: %s",
                model_id
            )

            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )

            # CPU inference
            pipeline = pipeline.to("cpu")

            # Disable progress bars in backend logs
            pipeline.set_progress_bar_config(
                disable=True
            )

            _pipeline = pipeline

            logger.info(
                "Local image model loaded successfully."
            )

            return _pipeline

        except Exception as error:
            logger.exception(
                "Local image model loading failed."
            )

            raise RuntimeError(
                f"Could not load local image model: {error}"
            ) from error


def generate_local_image(prompt: str) -> dict:
    """
    Generate an image locally using a CPU-based Diffusers model.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Image prompt cannot be empty.")

    prompt = prompt.strip()

    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Prompt cannot exceed {MAX_PROMPT_LENGTH} characters."
        )

    filename = f"{uuid.uuid4().hex}_local.png"
    image_path = IMAGE_DIR / filename

    model_id = os.getenv(
        "LOCAL_IMAGE_MODEL",
        DEFAULT_LOCAL_MODEL
    )

    try:
        pipeline = get_local_pipeline()

        # Avoid concurrent CPU inference.
        with _generation_lock:

            generated = pipeline(
                prompt=prompt,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=0.0,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH
            )

        if not generated.images:
            raise RuntimeError(
                "Local model returned no image."
            )

        image = generated.images[0]

        IMAGE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        image.save(
            image_path,
            format="PNG"
        )

        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise RuntimeError(
                "Generated image could not be saved correctly."
            )

        logger.info(
            "Local image generated successfully: %s",
            filename
        )

        return {
            "filename": filename,
            "image_path": str(image_path),
            "prompt": prompt,
            "provider": "local",
            "model": model_id,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "status": "success"
        }

    except Exception as error:

        if image_path.exists():
            image_path.unlink()

        logger.exception(
            "Local image generation failed."
        )

        raise RuntimeError(
            f"Local image generation failed: {error}"
        ) from error
