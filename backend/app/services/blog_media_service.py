
"""
NEXORA AI - Blog Media Integration

Finds image placeholders in generated blogs,
generates images and replaces placeholders with Markdown images.
"""

import logging
import re

from app.services.image_generation_service import generate_cloud_image


logger = logging.getLogger(__name__)

MAX_IMAGES_PER_BLOG = 3

IMAGE_PATTERN = re.compile(
    r"<!--\s*NEXORA_IMAGE:\s*(.*?)\s*-->",
    re.IGNORECASE | re.DOTALL
)


def add_images_to_blog(blog_content: str) -> dict:
    """
    Generate images for blog placeholders and embed their URLs.
    Maximum three images per blog.
    """

    if not blog_content or not blog_content.strip():
        raise ValueError("Blog content cannot be empty.")

    matches = list(IMAGE_PATTERN.finditer(blog_content))

    if not matches:
        return {
            "content": blog_content,
            "images": [],
            "image_count": 0
        }

    updated_content = blog_content
    generated_images = []

    for match in matches[:MAX_IMAGES_PER_BLOG]:

        image_prompt = match.group(1).strip()

        if not image_prompt:
            continue

        try:
            result = generate_cloud_image(image_prompt)

            filename = result["filename"]

            image_url = (
                f"/api/creative/images/{filename}"
            )

            markdown_image = (
                f"![Generated image]({image_url})"
            )

            updated_content = updated_content.replace(
                match.group(0),
                markdown_image,
                1
            )

            generated_images.append({
                "filename": filename,
                "image_url": image_url,
                "prompt": image_prompt
            })

        except Exception as error:

            logger.warning(
                "Blog image generation failed: %s",
                error
            )

            # Preserve the original placeholder.
            continue

    return {
        "content": updated_content,
        "images": generated_images,
        "image_count": len(generated_images)
    }
