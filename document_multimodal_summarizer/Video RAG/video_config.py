# -*- coding: utf-8 -*-
"""
video_config.py
---------------
Shared configuration for the Video RAG pipeline (LlamaIndex-based).

API Key and paths are loaded from the `.env` file at the project root.
NEVER hardcode secrets or URLs here.

System Requirements:
    - FFmpeg installed and in PATH
      Windows installer: https://ffmpeg.org/download.html
      Or via winget: winget install ffmpeg

Note:
    Uses the same GOOGLE_API_KEY as the parent project (config.py).
    VIDEO_URL is NOT stored here — it is provided at runtime by the
    caller (webapp or direct function call).
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load .env from project root  (three levels up from Video RAG/)
# ------------------------------------------------------------------
_THIS_FILE    = os.path.abspath(__file__)
_VIDEO_DIR    = os.path.dirname(_THIS_FILE)                            # Video RAG/
_SRC_DIR      = os.path.dirname(_VIDEO_DIR)                           # Document Multimodal Summrizer/
_PROJECT_ROOT = os.path.normpath(os.path.join(_SRC_DIR, ".."))        # Multi_modal_RAG/

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ------------------------------------------------------------------
# Google Gemini API Key  (loaded from .env — never hardcoded)
# ------------------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. "
        "Create a .env file at the project root with: GOOGLE_API_KEY=<your_key>"
    )
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
print("Video RAG config: Google API key configured.")

# ------------------------------------------------------------------
# FFmpeg — Windows Note
# Install FFmpeg manually and ensure it is on your PATH.
# Optionally set explicit path below:
# os.environ["PATH"] += r";C:\path\to\ffmpeg\bin"
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Video / Audio / Output Paths  (relative to working directory)
# ------------------------------------------------------------------
# VIDEO_URL is intentionally NOT defined here.
# It is provided at runtime by the caller (user input / web UI).

# All paths are absolute (anchored to _VIDEO_DIR) so they resolve correctly
# regardless of which directory uvicorn / the caller is launched from.
OUTPUT_VIDEO_DIR  = os.path.join(_VIDEO_DIR, "video_data")
OUTPUT_MIXED_DIR  = os.path.join(_VIDEO_DIR, "mixed_data")
OUTPUT_AUDIO_PATH = os.path.join(OUTPUT_MIXED_DIR, "output_audio.wav")
VIDEO_FILE_PATH   = os.path.join(OUTPUT_VIDEO_DIR, "input_vid.mp4")

# LanceDB storage directory (absolute)
LANCEDB_URI = os.path.join(_VIDEO_DIR, "lancedb")

# ------------------------------------------------------------------
# Video indexing / extraction tuning
# ------------------------------------------------------------------
FRAME_EXTRACTION_FPS = 0.05  # 1 frame every 20 seconds
MAX_INDEXED_FRAMES   = 12

# ------------------------------------------------------------------
# Create required output directories on import
# ------------------------------------------------------------------
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_MIXED_DIR, exist_ok=True)

print(f"Working directory : {os.getcwd()}")
print(f"Video output dir  : {OUTPUT_VIDEO_DIR}")
print(f"Mixed data dir    : {OUTPUT_MIXED_DIR}")
