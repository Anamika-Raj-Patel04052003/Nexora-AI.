
import os
import base64
import logging
import tempfile
import subprocess

from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.8-27b"
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


# --------------------------------------------------
# GROQ CLIENT
# --------------------------------------------------

def get_groq_client():

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing from the .env file."
        )

    return Groq(api_key=GROQ_API_KEY)


# --------------------------------------------------
# IMAGE VALIDATION
# --------------------------------------------------

def validate_image(
    image_bytes: bytes,
    content_type: str
):

    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError(
            "Image size must not exceed 10 MB."
        )

    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(
            "Unsupported image format. "
            "Upload JPG, PNG, or WEBP."
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()

    except (UnidentifiedImageError, OSError):

        raise ValueError(
            "Uploaded file is not a valid image."
        )

    return True


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

def analyze_image(
    image_bytes: bytes,
    content_type: str,
    question: str
) -> str:

    question = question.strip()

    if not question:
        question = "Describe this image in detail."

    # Validate image before sending it to the AI model
    validate_image(
        image_bytes=image_bytes,
        content_type=content_type
    )

    # Convert image bytes into Base64
    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    image_data_url = (
        f"data:{content_type};base64,{encoded_image}"
    )

    try:
        client = get_groq_client()

        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,

            messages=[
                {
                    "role": "user",

                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],

            temperature=0.2,
            max_completion_tokens=1500
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError(
                "Vision model returned an empty response."
            )

        return answer.strip()

    except Exception:
        logger.exception("Image analysis failed.")
        raise


# --------------------------------------------------
# AUDIO CONFIGURATION
# --------------------------------------------------

GROQ_AUDIO_MODEL = os.getenv(
    "GROQ_AUDIO_MODEL",
    "whisper-large-v3-turbo"
)

MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".webm"
}


# --------------------------------------------------
# AUDIO VALIDATION
# --------------------------------------------------

def validate_audio(
    audio_bytes: bytes,
    filename: str
):

    if not audio_bytes:
        raise ValueError("Uploaded audio file is empty.")

    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise ValueError(
            "Audio file size must not exceed 25 MB."
        )

    extension = os.path.splitext(filename)[1].lower()

    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            "Unsupported audio format. "
            "Upload MP3, WAV, M4A, OGG, FLAC or WEBM."
        )

    return True


# --------------------------------------------------
# AUDIO TRANSCRIPTION
# --------------------------------------------------

def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    language: str | None = None
) -> str:

    validate_audio(
        audio_bytes=audio_bytes,
        filename=filename
    )

    try:
        client = get_groq_client()

        transcription = client.audio.transcriptions.create(
            file=(
                filename,
                audio_bytes
            ),

            model=GROQ_AUDIO_MODEL,
            response_format="json",
            temperature=0.0,
            language=language if language else None
        )

        transcript = transcription.text.strip()

        if not transcript:
            raise ValueError(
                "No speech could be recognized in the audio."
            )

        return transcript

    except Exception:
        logger.exception("Audio transcription failed.")
        raise


# --------------------------------------------------
# VIDEO CONFIGURATION
# --------------------------------------------------

GROQ_VIDEO_MODEL = os.getenv(
    "GROQ_VIDEO_MODEL",
    GROQ_VISION_MODEL
)

MAX_VIDEO_SIZE = 25 * 1024 * 1024  # 25 MB

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm"
}

MAX_VIDEO_FRAMES = 3


# --------------------------------------------------
# VIDEO VALIDATION
# --------------------------------------------------

def validate_video(
    video_bytes: bytes,
    filename: str
):

    if not video_bytes:
        raise ValueError("Uploaded video is empty.")

    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise ValueError(
            "Video size must not exceed 25 MB."
        )

    extension = os.path.splitext(filename)[1].lower()

    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            "Unsupported video format. "
            "Upload MP4 or WEBM."
        )

    return True


# --------------------------------------------------
# EXTRACT FRAMES FROM VIDEO
# --------------------------------------------------

def extract_video_frames(
    video_bytes: bytes,
    filename: str,
    max_frames: int = MAX_VIDEO_FRAMES
) -> list[bytes]:

    import cv2

    validate_video(
        video_bytes=video_bytes,
        filename=filename
    )

    extension = os.path.splitext(filename)[1].lower()

    temp_path = None
    frames = []
    capture = None

    try:

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:

            temp_file.write(video_bytes)
            temp_path = temp_file.name

        # Open video using OpenCV
        capture = cv2.VideoCapture(temp_path)

        if not capture.isOpened():
            raise ValueError(
                "Unable to open the uploaded video."
            )

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = capture.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0 or fps <= 0:
            raise ValueError(
                "Invalid video or video metadata."
            )

        # Calculate evenly distributed frame positions
        frame_count = min(
            max_frames,
            total_frames
        )

        positions = [
            round(
                index * (total_frames - 1)
                / max(frame_count - 1, 1)
            )
            for index in range(frame_count)
        ]

        # Extract selected frames
        for position in positions:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                position
            )

            success, frame = capture.read()

            if not success:
                continue

            # Resize large frames
            height, width = frame.shape[:2]

            max_dimension = 1024

            if max(height, width) > max_dimension:

                scale = max_dimension / max(
                    height,
                    width
                )

                new_width = int(width * scale)
                new_height = int(height * scale)

                frame = cv2.resize(
                    frame,
                    (new_width, new_height)
                )

            # Convert frame to JPEG bytes
            success, encoded = cv2.imencode(
                ".jpg",
                frame
            )

            if success:
                frames.append(
                    encoded.tobytes()
                )

        if not frames:
            raise ValueError(
                "No frames could be extracted from the video."
            )

        return frames

    finally:

        if capture is not None:
            capture.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# VIDEO AUDIO EXTRACTION
# --------------------------------------------------

def extract_video_audio(
    video_bytes: bytes,
    filename: str
) -> bytes:

    import imageio_ffmpeg

    validate_video(
        video_bytes=video_bytes,
        filename=filename
    )

    input_path = None
    output_path = None

    try:

        suffix = Path(filename).suffix.lower()

        # Create temporary input video
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as input_file:

            input_file.write(video_bytes)
            input_path = input_file.name

        # Create temporary WAV output
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as output_file:

            output_path = output_file.name

        # Get FFmpeg executable path
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_path,
            "-y",
            "-i", input_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            output_path
        ]

        # Execute FFmpeg
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise ValueError(
                "Audio extraction failed. "
                "Check whether the video contains an audio track."
            )

        # Read extracted audio
        with open(output_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        if not audio_bytes:
            raise ValueError(
                "No audio data found in the video."
            )

        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise ValueError(
                "Extracted audio is larger than "
                "the supported 25 MB limit."
            )

        return audio_bytes

    finally:

        # Remove temporary files
        for path in [input_path, output_path]:

            if path and os.path.exists(path):
                os.remove(path)


# --------------------------------------------------
# VIDEO UNDERSTANDING
# --------------------------------------------------

def analyze_video(
    video_bytes: bytes,
    filename: str,
    question: str,
    language: str | None = None
) -> dict:

    question = question.strip()

    if not question:
        question = "Summarize this video."

    # Validate video
    validate_video(
        video_bytes=video_bytes,
        filename=filename
    )

    try:

        # ------------------------------------------
        # 1. EXTRACT VIDEO FRAMES
        # ------------------------------------------

        frames = extract_video_frames(
            video_bytes=video_bytes,
            filename=filename
        )

        # ------------------------------------------
        # 2. EXTRACT AND TRANSCRIBE VIDEO AUDIO
        # ------------------------------------------

        transcript = ""

        try:

            audio_bytes = extract_video_audio(
                video_bytes=video_bytes,
                filename=filename
            )

            transcript = transcribe_audio(
                audio_bytes=audio_bytes,
                filename="video_audio.wav",
                language=language
            )

        except Exception:

            logger.warning(
                "Video audio transcription unavailable. "
                "Continuing with visual analysis.",
                exc_info=True
            )

            transcript = "No audio transcript was available."

        # ------------------------------------------
        # 3. PREPARE MULTIMODAL INPUT
        # ------------------------------------------

        content = [
            {
                "type": "text",

                "text": (
                    "Analyze the following video using "
                    "its extracted frames and audio transcript.\n\n"

                    f"User question: {question}\n\n"

                    f"Audio transcript:\n{transcript}\n\n"

                    "Instructions:\n"
                    "1. Describe the important visual events.\n"
                    "2. Use the transcript when relevant.\n"
                    "3. Connect visual events with spoken content.\n"
                    "4. Answer the user's question accurately.\n"
                    "5. Do not invent events or dialogue.\n"
                    "6. Mention when information is unavailable.\n"
                    "7. Respond in the user's language."
                )
            }
        ]

        # ------------------------------------------
        # 4. ATTACH EXTRACTED VIDEO FRAMES
        # ------------------------------------------

        for index, frame_bytes in enumerate(frames):

            encoded_frame = base64.b64encode(
                frame_bytes
            ).decode("utf-8")

            image_data_url = (
                "data:image/jpeg;base64,"
                f"{encoded_frame}"
            )

            content.append(
                {
                    "type": "text",
                    "text": f"Video frame {index + 1}:"
                }
            )

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                }
            )

        # ------------------------------------------
        # 5. SEND TO GROQ VISION MODEL
        # ------------------------------------------

        client = get_groq_client()

        response = client.chat.completions.create(
            model=GROQ_VIDEO_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],

            temperature=0.2,
            max_completion_tokens=2500
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError(
                "Video analysis returned an empty response."
            )

        return {
            "answer": answer.strip(),
            "transcript": transcript,
            "frames_analyzed": len(frames)
        }

    except Exception:

        logger.exception("Video analysis failed.")
        raise


