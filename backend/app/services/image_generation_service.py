"""
NEXORA AI - Image Generation Service

Multi-tier Cloud-based Text-to-Image Generation:
1. Pollinations.ai (Free standard endpoint without paid flux flags)
2. Hugging Face Serverless (Using stable-diffusion-xl-base-1.0 to avoid fal-ai 403)
3. Smart Canvas Fallback: Guarantees image & video generation NEVER crashes!
"""

import io
import logging
import os
import random
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Backend directory
BACKEND_DIR = Path(__file__).resolve().parents[2]

# Generated image storage
IMAGE_DIR = BACKEND_DIR / "generated" / "images"

# Standard Serverless Free Model (does not trigger fal-ai 403 Forbidden)
DEFAULT_HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
IMAGE_TIMEOUT = 45
MAX_PROMPT_LENGTH = 1000


def _generate_via_pollinations(prompt: str, image_path: Path) -> dict:
    """
    Generate image using Pollinations.ai free public tier.
    NOTE: We do NOT pass &model=flux because flux requires paid pollen credits (HTTP 402).
    """
    seed = random.randint(1000, 9999999)
    encoded_prompt = urllib.parse.quote(prompt.strip())
    
    # Endpoints without paid model tags
    urls = [
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={IMAGE_WIDTH}&height={IMAGE_HEIGHT}&nologo=true&seed={seed}",
        f"https://gen.pollinations.ai/image/{encoded_prompt}?width={IMAGE_WIDTH}&height={IMAGE_HEIGHT}&nologo=true&seed={seed}"
    ]

    last_error = None
    for url in urls:
        try:
            logger.info("Calling Pollinations: %s", url[:80])
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "image/png,image/jpeg,image/*"
                }
            )

            IMAGE_DIR.mkdir(parents=True, exist_ok=True)

            with urllib.request.urlopen(req, timeout=IMAGE_TIMEOUT) as response:
                if response.status == 200:
                    data = response.read()
                    if data and len(data) > 1000:
                        image = Image.open(io.BytesIO(data))
                        image.save(image_path, format="PNG")
                        return {
                            "provider": "pollinations",
                            "model": "standard"
                        }
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Pollinations endpoints failed: {last_error}")


def _generate_via_huggingface(prompt: str, image_path: Path, token: str) -> dict:
    """
    Generate image using Hugging Face Serverless Inference.
    Uses stabilityai/stable-diffusion-xl-base-1.0 to avoid fal-ai commercial router 403.
    """
    from huggingface_hub import InferenceClient

    model = os.getenv("HF_IMAGE_MODEL") or DEFAULT_HF_MODEL
    logger.info("Attempting Hugging Face image generation using: %s", model)

    client = InferenceClient(
        model=model,
        token=token,
        timeout=IMAGE_TIMEOUT
    )

    generated_image = client.text_to_image(
        prompt=prompt,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT
    )

    if generated_image is None:
        raise RuntimeError("Hugging Face returned an empty image.")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    generated_image.save(image_path, format="PNG")

    return {
        "provider": "huggingface",
        "model": model
    }


def _generate_fallback_canvas(prompt: str, image_path: Path) -> dict:
    """
    Local visual slide generator. Runs 100% locally with zero internet/keys required.
    Guarantees that video compilation and creative services NEVER fail!
    """
    logger.info("Creating local aesthetic slide canvas for: '%s'", prompt[:60])
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        [30, 30, IMAGE_WIDTH - 30, IMAGE_HEIGHT - 30],
        radius=24,
        outline=(99, 102, 241),
        width=3
    )
    draw.rounded_rectangle(
        [50, 50, IMAGE_WIDTH - 50, IMAGE_HEIGHT - 50],
        radius=20,
        fill=(30, 41, 59)
    )

    draw.line([(80, 160), (IMAGE_WIDTH - 80, 160)], fill=(129, 140, 248), width=3)

    title_text = "NEXORA AI  •  CREATIVE STUDIO"
    draw.text((IMAGE_WIDTH // 2, 110), title_text, fill=(165, 180, 252), anchor="mm")

    clean_p = prompt.strip()
    if len(clean_p) > 120:
        clean_p = clean_p[:117] + "..."
    draw.text((IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2), f"Scene:\n{clean_p}", fill=(248, 250, 252), anchor="mm", align="center")

    img.save(image_path, format="PNG")

    return {
        "provider": "local_canvas",
        "model": "nexora-canvas"
    }


def generate_cloud_image(prompt: str) -> dict:
    """
    Unified Image Generation:
    1. Tries Pollinations free public tier (no model=flux to prevent 402)
    2. Tries Hugging Face SDXL (without commercial fal-ai to prevent 403)
    3. Seamless Local Aesthetic Canvas fallback so video/blog NEVER fails!
    """
    if not prompt or not prompt.strip():
        raise ValueError("Image prompt cannot be empty.")

    prompt = prompt.strip()

    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Image prompt cannot exceed {MAX_PROMPT_LENGTH} characters."
        )

    filename = f"{uuid.uuid4().hex}.png"
    image_path = IMAGE_DIR / filename

    token = os.getenv("HF_TOKEN")
    generation_info = None

    # Step 1: Pollinations standard endpoint (free, no 402)
    try:
        generation_info = _generate_via_pollinations(prompt, image_path)
    except Exception as p_err:
        logger.warning("Pollinations image generation bypassed: %s", p_err)

    # Step 2: Hugging Face (if token present and Pollinations failed)
    if not generation_info and token:
        try:
            generation_info = _generate_via_huggingface(prompt, image_path, token)
        except Exception as hf_err:
            logger.warning("Hugging Face image generation bypassed: %s", hf_err)

    # Step 3: Absolute local fallback if both remote APIs had issues
    if not generation_info or not image_path.is_file() or image_path.stat().st_size == 0:
        generation_info = _generate_fallback_canvas(prompt, image_path)

    logger.info("Image ready via: %s", generation_info["provider"])

    return {
        "filename": filename,
        "image_path": str(image_path),
        "prompt": prompt,
        "provider": generation_info["provider"],
        "model": generation_info["model"],
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "status": "success"
    }