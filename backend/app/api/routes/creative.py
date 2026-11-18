"""
NEXORA AI - Creative Studio API Routes

Provides endpoints for:
1. Blog generation (with LaTeX formulas, comparison tables, inline images, audio narration & video slideshow)
2. Standalone image generation (Cloud FLUX & Local Diffusers)
3. Standalone audio generation (Edge-TTS)
4. Video slideshow compilation (MoviePy)
5. Media retrieval endpoints (Images, Audio, Video)
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
# COMMON VALIDATION UTILITIES
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
            detail=f"Invalid filename: must be a plain file ending with {extension}"
        )


def get_media_file(
    directory: Path,
    filename: str,
    extension: str,
    media_type: str
) -> FileResponse:
    """
    Validate and stream a generated media file.
    """
    validate_filename(filename, extension)
    file_path = directory / filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Requested media file '{filename}' was not found on the server."
        )

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


def handle_service_error(error: Exception, operation: str) -> None:
    """
    Log internal errors with stack traces.
    """
    logger.exception("%s failed: %s", operation, error)
    raise HTTPException(
        status_code=500,
        detail=f"{operation} encountered an internal error: {str(error)}"
    ) from error


# ==================================================
# 1. ENHANCED BLOG GENERATION (+ IMAGES + AUDIO + VIDEO)
# ==================================================

class BlogRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Main topic or prompt for the blog"
    )
    language: str = Field(
        default="English",
        min_length=2,
        max_length=50,
        description="Target language (English, Hindi, Hinglish, etc.)"
    )
    tone: str = Field(
        default="Professional",
        min_length=2,
        max_length=50,
        description="Writing tone (Professional, Engaging, Academic, Creative, etc.)"
    )
    length: Literal["Short", "Medium", "Long"] = Field(
        default="Medium",
        description="Length of the blog"
    )
    include_images: bool = Field(
        default=True,
        description="Automatically generate and embed in-between AI images"
    )
    image_mode: Literal["cloud", "local"] = Field(
        default="cloud",
        description="Image generation provider ('cloud' using FLUX.1 or 'local' using Tiny-SD)"
    )
    include_audio: bool = Field(
        default=False,
        description="Generate an Edge-TTS audio narration podcast of the blog"
    )
    voice: str = Field(
        default=DEFAULT_VOICE,
        min_length=3,
        max_length=100,
        description="Edge-TTS voice selection"
    )
    include_video: bool = Field(
        default=False,
        description="Generate a complete MoviePy MP4 slideshow combining generated images and audio"
    )


class BlogResponse(BaseModel):
    topic: str
    content: str
    status: str
    images: list[dict] = Field(default_factory=list)
    image_count: int = 0
    audio: dict | None = None
    audio_url: str | None = None
    video: dict | None = None
    video_url: str | None = None


@router.post("/blog", response_model=BlogResponse)
async def create_blog(request: BlogRequest):
    """
    Generate an in-depth, publication-ready blog with:
    - LaTeX formulas ($...$ inline, $$...$$ block)
    - Markdown comparison tables
    - Auto-generated in-between images
    - Optional Edge-TTS audio narration
    - Optional MoviePy MP4 video slideshow
    """
    topic_clean = request.topic.strip()
    if not topic_clean:
        raise HTTPException(
            status_code=400,
            detail="Blog topic cannot be empty."
        )

    try:
        # Step 1: Generate rich structured blog content with formulas & tables
        logger.info("Generating blog content for topic: '%s'", topic_clean[:60])
        blog_content = generate_blog(
            topic=topic_clean,
            language=request.language.strip(),
            tone=request.tone.strip(),
            length=request.length
        )

        images = []
        image_count = 0
        audio_data = None
        audio_url = None
        video_data = None
        video_url = None

        # Step 2: Auto-generate in-between images if requested
        if request.include_images:
            logger.info("Processing in-between blog images (mode: %s)...", request.image_mode)
            try:
                media_result = add_images_to_blog(
                    blog_content=blog_content,
                    mode=request.image_mode
                )
                blog_content = media_result["content"]
                images = media_result["images"]
                image_count = media_result["image_count"]
            except Exception as img_err:
                logger.warning("Blog image generation partially failed: %s", img_err)

        # Step 3: Auto-generate narration audio if requested
        audio_filename = None
        if request.include_audio:
            logger.info("Generating blog audio narration using voice: %s...", request.voice)
            try:
                audio_result = await generate_blog_audio(
                    blog_content=blog_content,
                    voice=request.voice.strip()
                )
                audio_filename = audio_result["filename"]
                audio_data = {
                    "filename": audio_filename,
                    "voice": audio_result["voice"],
                    "provider": audio_result["provider"],
                    "chunk_count": audio_result["chunk_count"],
                    "character_count": audio_result["character_count"]
                }
                audio_url = f"/api/creative/audio/{audio_filename}"
            except Exception as aud_err:
                logger.warning("Blog audio narration failed: %s", aud_err)

        # Step 4: Auto-generate video slideshow if requested (requires images and audio)
        if request.include_video:
            if not images:
                logger.warning("Video requested but no images were generated.")
            elif not audio_filename:
                logger.warning("Video requested but audio narration was not enabled or failed.")
            else:
                logger.info("Creating MP4 video slideshow from %d images and narration audio...", len(images))
                try:
                    image_filenames = [img["filename"] for img in images if "filename" in img]
                    video_result = create_local_video(
                        image_filenames=image_filenames,
                        audio_filename=audio_filename
                    )
                    video_filename = video_result["filename"]
                    video_data = {
                        "filename": video_filename,
                        "duration": video_result["duration"],
                        "image_count": video_result["image_count"],
                        "resolution": video_result.get("resolution"),
                        "fps": video_result.get("fps")
                    }
                    video_url = f"/api/creative/videos/{video_filename}"
                except Exception as vid_err:
                    logger.warning("Video compilation failed: %s", vid_err)

        return BlogResponse(
            topic=topic_clean,
            content=blog_content,
            status="success",
            images=images,
            image_count=image_count,
            audio=audio_data,
            audio_url=audio_url,
            video=video_data,
            video_url=video_url
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        handle_service_error(error, "Blog creation")


# ==================================================
# 2. STANDALONE IMAGE GENERATION
# ==================================================

class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1000)
    mode: Literal["cloud", "local"] = "cloud"


@router.post("/image")
def create_image(request: ImageRequest):
    """
    Generate an image standalone from a prompt.
    """
    prompt_clean = request.prompt.strip()
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="Image prompt cannot be empty.")

    try:
        result = generate_image(
            prompt=prompt_clean,
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
            "image_url": f"/api/creative/images/{result['filename']}"
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        handle_service_error(error, "Image generation")


# --------------------------------------------------
# GENERATED IMAGE FILE RETRIEVAL
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
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default=DEFAULT_VOICE, min_length=3, max_length=100)


@router.post("/audio")
async def create_audio(request: AudioRequest):
    """
    Generate an MP3 audio file from text using Edge-TTS.
    """
    text_clean = request.text.strip()
    if not text_clean:
        raise HTTPException(status_code=400, detail="Audio text cannot be empty.")

    try:
        result = await generate_audio(
            text=text_clean,
            voice=request.voice.strip()
        )

        return {
            "status": "success",
            "filename": result["filename"],
            "voice": result["voice"],
            "provider": result["provider"],
            "audio_url": f"/api/creative/audio/{result['filename']}"
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        handle_service_error(error, "Audio generation")


# --------------------------------------------------
# GENERATED AUDIO FILE RETRIEVAL
# --------------------------------------------------

@router.get("/audio/{filename}")
def get_generated_audio(filename: str):
    from app.services.audio_generation_config import AUDIO_DIR

    return get_media_file(
        directory=AUDIO_DIR,
        filename=filename,
        extension=".mp3",
        media_type="audio/mpeg"
    )


# ==================================================
# 4. STANDALONE VIDEO COMPILATION
# ==================================================

class VideoRequest(BaseModel):
    image_filenames: list[str] = Field(..., min_length=1, max_length=20)
    audio_filename: str = Field(..., min_length=1, max_length=150)


@router.post("/video")
def create_video(request: VideoRequest):
    """
    Compile multiple PNG images and an MP3 audio into an MP4 slideshow video.
    """
    try:
        for filename in request.image_filenames:
            validate_filename(filename, ".png")

        validate_filename(request.audio_filename, ".mp3")

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
            "video_url": f"/api/creative/videos/{result['filename']}"
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        handle_service_error(error, "Video creation")


# --------------------------------------------------
# GENERATED VIDEO FILE RETRIEVAL
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