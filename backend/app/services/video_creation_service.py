
"""
NEXORA AI - Local Video Creation Service

Combines generated images and audio into an MP4 slideshow.
"""

import logging
import math
import uuid

from pathlib import Path

from moviepy import (
    ImageClip,
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    concatenate_videoclips
)

from app.services.image_generation_service import IMAGE_DIR
from app.services.audio_generation_config import AUDIO_DIR


logger = logging.getLogger(__name__)


BACKEND_DIR = Path(__file__).resolve().parents[2]

VIDEO_DIR = BACKEND_DIR / "generated" / "videos"

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 24

MAX_IMAGES = 20


def validate_filename(filename: str, extension: str) -> None:
    """Validate filenames and prevent directory traversal."""

    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "\x00" in filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or not filename.lower().endswith(extension.lower())
    ):
        raise ValueError("Invalid filename.")


def create_local_video(
    image_filenames: list[str],
    audio_filename: str
) -> dict:
    """
    Create an MP4 slideshow using generated images and audio.
    """

    # ----------------------------------------------
    # INPUT VALIDATION
    # ----------------------------------------------

    if not isinstance(image_filenames, list) or not image_filenames:
        raise ValueError("At least one image is required.")

    if len(image_filenames) > MAX_IMAGES:
        raise ValueError(
            f"Maximum {MAX_IMAGES} images are allowed."
        )

    validate_filename(audio_filename, ".mp3")

    # ----------------------------------------------
    # IMAGE VALIDATION
    # ----------------------------------------------

    image_paths = []

    for image_filename in image_filenames:

        validate_filename(image_filename, ".png")

        image_path = IMAGE_DIR / image_filename

        if not image_path.is_file():
            raise ValueError(
                f"Image not found: {image_filename}"
            )

        image_paths.append(image_path)

    # ----------------------------------------------
    # AUDIO VALIDATION
    # ----------------------------------------------

    audio_path = AUDIO_DIR / audio_filename

    if not audio_path.is_file():
        raise ValueError("Audio file not found.")

    # ----------------------------------------------
    # OUTPUT CONFIGURATION
    # ----------------------------------------------

    VIDEO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = f"{uuid.uuid4().hex}.mp4"
    output_path = VIDEO_DIR / filename

    clips = []
    resources = []

    audio = None
    final_video = None
    success = False

    try:

        # ------------------------------------------
        # LOAD AUDIO
        # ------------------------------------------

        audio = AudioFileClip(str(audio_path))

        if (
            audio.duration is None
            or not math.isfinite(audio.duration)
            or audio.duration <= 0
        ):
            raise ValueError(
                "Audio duration must be greater than zero."
            )

        duration_per_image = audio.duration / len(image_paths)

        # ------------------------------------------
        # CREATE IMAGE SLIDES
        # ------------------------------------------

        for image_path in image_paths:

            image_clip = ImageClip(str(image_path))

            resources.append(image_clip)

            # Maintain original aspect ratio.
            scale = min(
                VIDEO_WIDTH / image_clip.w,
                VIDEO_HEIGHT / image_clip.h
            )

            new_width = max(
                1,
                int(image_clip.w * scale)
            )

            new_height = max(
                1,
                int(image_clip.h * scale)
            )

            image_clip = image_clip.resized(
                new_size=(new_width, new_height)
            )

            image_clip = image_clip.with_duration(
                duration_per_image
            )

            # Black background for 16:9 video.
            background = ColorClip(
                size=(VIDEO_WIDTH, VIDEO_HEIGHT),
                color=(0, 0, 0)
            ).with_duration(duration_per_image)

            resources.append(background)

            # Center the image without stretching.
            composed_clip = CompositeVideoClip(
                [
                    background,
                    image_clip.with_position("center")
                ],
                size=(VIDEO_WIDTH, VIDEO_HEIGHT)
            ).with_duration(duration_per_image)

            clips.append(composed_clip)

        # ------------------------------------------
        # COMBINE VIDEO CLIPS
        # ------------------------------------------

        final_video = concatenate_videoclips(
            clips,
            method="compose"
        )

        final_video = final_video.with_audio(audio)

        # ------------------------------------------
        # EXPORT VIDEO
        # ------------------------------------------

        final_video.write_videofile(
            str(output_path),
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            logger=None
        )

        # ------------------------------------------
        # OUTPUT VALIDATION
        # ------------------------------------------

        if (
            not output_path.is_file()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Generated video file is empty."
            )

        success = True

        logger.info(
            "Video created successfully: %s",
            filename
        )

        return {
            "filename": filename,
            "video_path": str(output_path),
            "duration": round(audio.duration, 2),
            "image_count": len(image_paths),
            "resolution": f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
            "fps": VIDEO_FPS,
            "status": "success"
        }

    except ValueError:
        raise

    except Exception as error:

        logger.exception("Video creation failed.")

        raise RuntimeError(
            f"Video creation failed: {error}"
        ) from error

    finally:

        # ------------------------------------------
        # RESOURCE CLEANUP
        # ------------------------------------------

        if final_video is not None:
            final_video.close()

        for clip in clips:
            clip.close()

        for resource in resources:
            resource.close()

        if audio is not None:
            audio.close()

        # Remove incomplete output.
        if not success:

            try:
                if output_path.exists():
                    output_path.unlink()

            except OSError:

                logger.warning(
                    "Could not remove incomplete video: %s",
                    output_path
                )
