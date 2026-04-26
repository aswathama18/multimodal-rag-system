# -*- coding: utf-8 -*-
r"""
config.py
---------
Shared configuration for the Multimodal Summarizer pipeline.

API Key & paths are loaded from the `.env` file at the project root.
NEVER hardcode secrets here.

Requirements:
    pip install pytesseract unstructured-pytesseract python-dotenv

System Requirements:
    - Tesseract OCR installed at path defined in .env (TESSERACT_PATH)
    - Poppler binaries at path defined in .env (POPPLER_PATH)
"""

import os
import pytesseract
import unstructured_pytesseract
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load .env from project root (two levels up from this file)
# ------------------------------------------------------------------
_THIS_FILE  = os.path.abspath(__file__)
_SRC_DIR    = os.path.dirname(_THIS_FILE)                       # Document Multimodal Summrizer/
_PROJECT_ROOT = os.path.normpath(os.path.join(_SRC_DIR, "..")) # Multi_modal_RAG/

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ------------------------------------------------------------------
# Tesseract OCR Configuration
# ------------------------------------------------------------------
tesseract_bin = os.environ.get(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"   # fallback
)
pytesseract.pytesseract.tesseract_cmd                  = tesseract_bin
unstructured_pytesseract.pytesseract.tesseract_cmd     = tesseract_bin
os.environ["PATH"] = os.path.dirname(tesseract_bin) + os.pathsep + os.environ.get("PATH", "")

# ------------------------------------------------------------------
# Poppler Configuration (required for PDF rendering)
# ------------------------------------------------------------------
poppler_path = os.environ.get(
    "POPPLER_PATH",
    r"C:\Users\nikhi\OneDrive\Desktop\Release-25.12.0-0\poppler-25.12.0\Library\bin"  # fallback
)

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

# ------------------------------------------------------------------
# Google Gemini Model Names  (correct lowercase-hyphen API identifiers)
# ------------------------------------------------------------------
# Most capable — used for final RAG answer generation
GEMINI_PRO_MODEL        = "gemini-2.5-pro-preview-05-06"

# Fast vision model — used for image summarisation
GEMINI_FLASH_MODEL      = "gemini-2.0-flash"

# Lightest & cheapest — used for bulk text / table summarisation
GEMINI_FLASH_LITE_MODEL = "gemini-2.0-flash-lite"

# Google text-embedding model — used by Chroma for semantic search
EMBED_MODEL             = "models/embedding-001"

# ------------------------------------------------------------------
# Chroma Vector Store  (disk-persistent for deployment readiness)
# ------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stored at project root so it survives restarts and can be shared
# across future applications added to this project.
CHROMA_PERSIST_DIR     = os.path.normpath(os.path.join(_BASE_DIR, "..", "chroma_db"))
CHROMA_COLLECTION_NAME = "mm_rag"

# ------------------------------------------------------------------
# Debug / terminal-fallback output directory
# ------------------------------------------------------------------
# When running outside Jupyter, display.py saves rendered images here
# so you can verify the pipeline output visually.
DEBUG_IMAGE_DIR = os.path.normpath(os.path.join(_BASE_DIR, "..", "debug_images"))

print("Config loaded: Tesseract, Poppler, Gemini models, and Chroma config set successfully!")
