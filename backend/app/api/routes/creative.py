
"""
NEXORA AI - Creative Studio API Routes

Provides endpoints for:
1. Blog generation
2. Image generation
3. Audio generation
4. Video creation
5. Accessing generated media files
"""

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Services
from app.services.creative_service import generate_blog
from app.services.image_service import generate_image
from app.services.audio_generation_service import generate_audio
from app.services.blog_audio_service import generate_blog_audio
from app.services.video_creation_service import create_local_video
from app.services.blog_media_service import add_images_to_blog
from app.services.audio_generation_config import DEFAULT_VOICE


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/creative",
    tags=["Creative Studio"]
)


# ==================================================
# COMMON VALIDATION
# ==================================================

def validate_filename(filename: str, extension: str) -> None:
    """
    Allow only plain filenames with the expected extension.
    Prevent directory traversal and invalid filenames.
    """

    if (
        not filename
        or filename in {".", ".."}
        or "\x00" in filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or not filename.lower().endswith(extension.lower())
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )


def get_media_file(
    directory: Path,
    filename: str,
    extension: str,
    media_type: str
) -> FileResponse:
    """
    Validate and return a generated media file.
    """

    validate_filename(filename, extension)

    file_path = directory / filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Generated file not found."
        )

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


def handle_service_error(
    error: Exception,
    operation: str
) -> None:
    """
    Log internal errors without exposing implementation details.
    """

    logger.exception("%s failed.", operation)

    raise HTTPException(
        status_code=500,
        detail=f"{operation} failed. Please try again later."
    ) from error


# ==================================================
# 1. BLOG GENERATION + OPTIONAL IMAGE + AUDIO
# ==================================================

class BlogRequest(BaseModel):

    topic: str = Field(
        min_length=3,
        max_length=500
    )

    language: str = Field(
        default="English",
        min_length=2,
        max_length=50
    )

    tone: str = Field(
        default="Professional",
        min_length=2,
        max_length=50
    )

    length: Literal["Short", "Medium", "Long"] = "Medium"

    include_images: bool = False
    include_audio: bool = False

    voice: str = Field(
        default=DEFAULT_VOICE,
        min_length=3,
        max_length=100
    )


class BlogResponse(BaseModel):

    topic: str
    content: str
    status: str

    images: list[dict] = Field(default_factory=list)
    image_count: int = 0

    audio: dict | None = None
    audio_url: str | None = None


@router.post("/blog", response_model=BlogResponse)
async def create_blog(request: BlogRequest):

    if not request.topic.strip():
        raise HTTPException(
            status_code=400,
            detail="Blog topic cannot be empty."
        )

    try:

        # Generate blog content
        blog_content = generate_blog(
            topic=request.topic.strip(),
            language=request.language.strip(),
            tone=request.tone.strip(),
            length=request.length
        )

        images = []
        image_count = 0
        audio_data = None
        audio_url = None

        # Optional image generation
        if request.include_images:

            media_result = add_images_to_blog(
                blog_content
            )

            blog_content = media_result["content"]
            images = media_result["images"]
            image_count = media_result["image_count"]

        # Optional blog narration
        if request.include_audio:

            audio_result = await generate_blog_audio(
                blog_content=blog_content,
                voice=request.voice.strip()
            )

            audio_data = {
                "filename": audio_result["filename"],
                "voice": audio_result["voice"],
                "provider": audio_result["provider"],
                "chunk_count": audio_result["chunk_count"],
                "character_count": audio_result["character_count"]
            }

            audio_url = (
                f"/api/creative/audio/{audio_result['filename']}"
            )

        return BlogResponse(
            topic=request.topic.strip(),
            content=blog_content,
            status="success",
            images=images,
            image_count=image_count,
            audio=audio_data,
            audio_url=audio_url
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except HTTPException:
        raise

    except Exception as error:

        handle_service_error(
            error,
            "Blog generation"
        )


# ==================================================
# 2. IMAGE GENERATION
# ==================================================

class ImageRequest(BaseModel):

    prompt: str = Field(
        min_length=3,
        max_length=1000
    )

    mode: Literal["cloud", "local"] = "cloud"


@router.post("/image")
def create_image(request: ImageRequest):

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Image prompt cannot be empty."
        )

    try:

        result = generate_image(
            prompt=request.prompt.strip(),
            mode=request.mode
        )

        return {
            "status": "success",
            "prompt": result["prompt"],
            "mode": result["mode"],
            "provider": result["provider"],
            "model": result["model"],
            "filename": result["filename"],
            "width": result.get("width"),
            "height": result.get("height"),
            "image_url": (
                f"/api/creative/images/{result['filename']}"
            )
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except HTTPException:
        raise

    except Exception as error:

        handle_service_error(
            error,
            "Image generation"
        )


# --------------------------------------------------
# GENERATED IMAGE ACCESS
# --------------------------------------------------

@router.get("/images/{filename}")
def get_generated_image(filename: str):

    from app.services.image_generation_service import IMAGE_DIR

    return get_media_file(
        directory=IMAGE_DIR,
        filename=filename,
        extension=".png",
        media_type="image/png"
    )


# ==================================================
# 3. STANDALONE AUDIO GENERATION
# ==================================================

class AudioRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=5000
    )

    voice: str = Field(
        default=DEFAULT_VOICE,
        min_length=3,
        max_length=100
    )


@router.post("/audio")
async def create_audio(request: AudioRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Audio text cannot be empty."
        )

    try:

        result = await generate_audio(
            text=request.text.strip(),
            voice=request.voice.strip()
        )

        return {
            "status": "success",
            "filename": result["filename"],
            "voice": result["voice"],
            "provider": result["provider"],
            "audio_url": (
                f"/api/creative/audio/{result['filename']}"
            )
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except HTTPException:
        raise

    except Exception as error:

        handle_service_error(
            error,
            "Audio generation"
        )


# --------------------------------------------------
# GENERATED AUDIO ACCESS
# --------------------------------------------------

@router.get("/audio/{filename}")
def get_generated_audio(filename: str):

    from app.services.audio_generation_service import AUDIO_DIR

    return get_media_file(
        directory=AUDIO_DIR,
        filename=filename,
        extension=".mp3",
        media_type="audio/mpeg"
    )


# ==================================================
# 4. VIDEO GENERATION
# ==================================================

class VideoRequest(BaseModel):

    image_filenames: list[str] = Field(
        min_length=1,
        max_length=20
    )

    audio_filename: str = Field(
        min_length=1,
        max_length=150
    )


@router.post("/video")
def create_video(request: VideoRequest):

    try:

        # Validate input filenames
        for filename in request.image_filenames:
            validate_filename(filename, ".png")

        validate_filename(
            request.audio_filename,
            ".mp3"
        )

        result = create_local_video(
            image_filenames=request.image_filenames,
            audio_filename=request.audio_filename
        )

        return {
            "status": "success",
            "filename": result["filename"],
            "duration": result["duration"],
            "image_count": result["image_count"],
            "resolution": result.get("resolution"),
            "fps": result.get("fps"),
            "video_url": (
                f"/api/creative/videos/{result['filename']}"
            )
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except HTTPException:
        raise

    except Exception as error:

        handle_service_error(
            error,
            "Video creation"
        )


# --------------------------------------------------
# GENERATED VIDEO ACCESS
# --------------------------------------------------

@router.get("/videos/{filename}")
def get_generated_video(filename: str):

    from app.services.video_creation_service import VIDEO_DIR

    return get_media_file(
        directory=VIDEO_DIR,
        filename=filename,
        extension=".mp4",
        media_type="video/mp4"
    )
