"""
NEXORA AI - Video Frame Extractor

Extracts evenly distributed, high-quality keyframes from uploaded videos
using OpenCV (cv2) for multimodal vision analysis:
- Video metadata inspection (total frames, FPS, duration)
- Uniform temporal sampling across video timeline
- Automatic high-resolution downscaling (max 1024px)
- JPEG byte encoding and memory-efficient cleanup
"""

import logging
import os
import tempfile
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

DEFAULT_MAX_FRAMES = 3
MAX_FRAME_DIMENSION = 1024


def extract_keyframes_from_video(
    video_bytes: bytes,
    filename: str,
    max_frames: int = DEFAULT_MAX_FRAMES
) -> list[bytes]:
    """
    Extracts up to max_frames evenly spaced frames from a video.

    Args:
        video_bytes: Raw binary video data.
        filename: Original filename (used to determine suffix).
        max_frames: Target number of frames to sample (default: 3).

    Returns:
        list[bytes]: List of JPEG-encoded image bytes.
    """
    if not video_bytes:
        raise ValueError("Video data is empty.")

    suffix = Path(filename).suffix.lower() or ".mp4"
    temp_path = None
    capture = None
    frames = []

    try:
        # Save video temporarily for OpenCV VideoCapture
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("Unable to open video stream with OpenCV.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0 or fps <= 0:
            raise ValueError("Video contains invalid or missing frame metadata.")

        duration_sec = total_frames / fps
        logger.info(
            "Video opened: %d frames, %.1f FPS, %.1fs duration. Sampling %d frames.",
            total_frames, fps, duration_sec, min(max_frames, total_frames)
        )

        # Calculate evenly spaced sampling positions
        target_count = min(max_frames, total_frames)
        positions = [
            round(i * (total_frames - 1) / max(target_count - 1, 1))
            for i in range(target_count)
        ]

        for idx, pos in enumerate(positions, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
            success, frame = capture.read()
            if not success or frame is None:
                continue

            # Scale down large frames for Vision LLM efficiency
            h, w = frame.shape[:2]
            if max(h, w) > MAX_FRAME_DIMENSION:
                scale = MAX_FRAME_DIMENSION / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode as JPEG
            encode_success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if encode_success:
                frames.append(buffer.tobytes())

        if not frames:
            raise ValueError("No readable frames could be extracted from the video.")

        logger.info("Successfully extracted %d keyframes from %s.", len(frames), filename)
        return frames

    finally:
        if capture is not None:
            capture.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass