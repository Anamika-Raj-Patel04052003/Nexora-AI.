"""
NEXORA AI - Blog Media Integration Service

Finds image placeholders (<!-- NEXORA_IMAGE: prompt -->) in generated blogs,
generates images via cloud (HuggingFace FLUX) or local (Diffusers) models,
and replaces placeholders with publication-ready Markdown figures with captions.
"""

import logging
import re
from typing import Literal

from app.services.image_service import generate_image


logger = logging.getLogger(__name__)

MAX_IMAGES_PER_BLOG = 3

IMAGE_PATTERN = re.compile(
    r"<!--\s*NEXORA_IMAGE:\s*(.*?)\s*-->",
    re.IGNORECASE | re.DOTALL
)


def add_images_to_blog(
    blog_content: str,
    mode: Literal["cloud", "local"] = "cloud"
) -> dict:
    """
    Scans the blog content for image placeholders, generates realistic images,
    and embeds them with responsive Markdown syntax and figure captions.

    Args:
        blog_content: The raw Markdown blog text containing placeholders.
        mode: Image generation provider ('cloud' using FLUX.1 or 'local' using Tiny-SD).

    Returns:
        dict: {
            "content": Updated Markdown string with embedded images,
            "images": List of generated image metadata,
            "image_count": Total number of successfully generated images
        }
    """

    if not isinstance(blog_content, str) or not blog_content.strip():
        raise ValueError("Blog content cannot be empty.")

    matches = list(IMAGE_PATTERN.finditer(blog_content))

    if not matches:
        logger.info("No image placeholders found in blog content.")
        return {
            "content": blog_content,
            "images": [],
            "image_count": 0
        }

    updated_content = blog_content
    generated_images = []

    # Limit to MAX_IMAGES_PER_BLOG (up to 3 images)
    for index, match in enumerate(matches[:MAX_IMAGES_PER_BLOG], start=1):
        raw_placeholder = match.group(0)
        image_prompt = match.group(1).strip()

        if not image_prompt:
            # Clean up empty placeholder
            updated_content = updated_content.replace(raw_placeholder, "", 1)
            continue

        try:
            logger.info(
                "Generating blog image %d/%d: '%s' (mode: %s)",
                index,
                min(len(matches), MAX_IMAGES_PER_BLOG),
                image_prompt[:60],
                mode
            )

            # Generate image via unified image service
            result = generate_image(prompt=image_prompt, mode=mode)
            filename = result["filename"]
            image_url = f"/api/creative/images/{filename}"

            # Prepare user-friendly caption
            caption = image_prompt
            if len(caption) > 90:
                caption = caption[:87] + "..."

            # Publication-grade Markdown figure with caption
            markdown_figure = (
                f"\n\n"
                f"![{caption}]({image_url})\n\n"
                f'<div align="center"><em>Figure {index}: {caption}</em></div>\n\n'
            )

            # Replace placeholder with Markdown figure
            updated_content = updated_content.replace(
                raw_placeholder,
                markdown_figure,
                1
            )

            generated_images.append({
                "figure_number": index,
                "filename": filename,
                "image_url": image_url,
                "prompt": image_prompt,
                "mode": mode,
                "provider": result.get("provider", "unknown")
            })

            logger.info(
                "Successfully embedded Figure %d: %s",
                index,
                filename
            )

        except Exception as error:
            logger.warning(
                "Failed to generate image for placeholder: '%s'. Error: %s",
                image_prompt[:60],
                error
            )
            # Remove failed placeholder to avoid leaving raw HTML comments in output
            updated_content = updated_content.replace(
                raw_placeholder,
                f"\n\n*(Visual representation: {image_prompt})*\n\n",
                1
            )
            continue

    # Clean up any excess placeholders beyond MAX_IMAGES_PER_BLOG
    if len(matches) > MAX_IMAGES_PER_BLOG:
        for excess_match in matches[MAX_IMAGES_PER_BLOG:]:
            updated_content = updated_content.replace(excess_match.group(0), "", 1)

    return {
        "content": updated_content.strip(),
        "images": generated_images,
        "image_count": len(generated_images)
    }