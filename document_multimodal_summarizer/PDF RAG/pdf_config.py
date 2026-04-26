# -*- coding: utf-8 -*-
r"""
pdf_config.py
-------------
Shared configuration for the PDF Multimodal RAG pipeline.

Handles:
  - Tesseract OCR binary path
  - Poppler binary path
  - Google Gemini API key (loaded from .env at project root)
  - Gemini model names (text, vision, embedding)
  - Chroma vectorstore persistence settings
  - Debug output directory

System Requirements
-------------------
  Tesseract OCR  →  https://github.com/UB-Mannheim/tesseract/wiki
  Poppler        →  https://github.com/oschwartz10612/poppler-windows/releases

Set the following in your .env file (project root):
  GOOGLE_API_KEY=<your_key>
  TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe   (optional override)
  POPPLER_PATH=C:\path\to\poppler\Library\bin                   (optional override)
"""

import os
import pytesseract
import unstructured_pytesseract
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Resolve project root from this file's location
# Folder layout:
#   Multi_modal_RAG/
#     Document Multimodal Summrizer/
#       PDF RAG/      <-- this file lives here
# ------------------------------------------------------------------
_THIS_FILE    = os.path.abspath(__file__)
_MODULE_DIR   = os.path.dirname(_THIS_FILE)            # PDF RAG/
_PARENT_DIR   = os.path.dirname(_MODULE_DIR)            # Document Multimodal Summrizer/
_PROJECT_ROOT = os.path.normpath(os.path.join(_PARENT_DIR, ".."))  # Multi_modal_RAG/

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ------------------------------------------------------------------
# Tesseract OCR
# ------------------------------------------------------------------
TESSERACT_BIN = os.environ.get(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",   # Windows default
)
pytesseract.pytesseract.tesseract_cmd              = TESSERACT_BIN
unstructured_pytesseract.pytesseract.tesseract_cmd = TESSERACT_BIN
os.environ["PATH"] = (
    os.path.dirname(TESSERACT_BIN) + os.pathsep + os.environ.get("PATH", "")
)

# ------------------------------------------------------------------
# Poppler  (required for PDF→image rendering in hi_res strategy)
# ------------------------------------------------------------------
POPPLER_PATH = os.environ.get(
    "POPPLER_PATH",
    r"C:\Users\nikhi\OneDrive\Desktop\Release-25.12.0-0\poppler-25.12.0\Library\bin",
)
os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

# ------------------------------------------------------------------
# Google Gemini API Key
# ------------------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. "
        "Create a .env file at the project root with: GOOGLE_API_KEY=<your_key>"
    )
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ------------------------------------------------------------------
# Gemini Model Names
# ------------------------------------------------------------------
# Final answer generation
GEMINI_PRO_MODEL        = "gemini-2.5-flash"

# Image summarisation (fast vision model)
GEMINI_FLASH_MODEL      = "gemini-2.5-flash"

# Table & text summarisation (lightest / cheapest)
GEMINI_FLASH_MODEL = "gemini-2.5-flash"

# Embedding model (used by Chroma for semantic search)
# Official Gemini embeddings docs now use gemini-embedding-001 for embedContent.
EMBED_MODEL             = "gemini-embedding-001"

# ------------------------------------------------------------------
# Text ingestion tuning
# ------------------------------------------------------------------
# The optimized PDF pipeline embeds raw text chunks directly instead of
# summarizing each text block with an LLM.
TEXT_CHUNK_SIZE            = 6000
TEXT_CHUNK_OVERLAP         = 200
TEXT_SECTION_MAX_CHARS     = 12000
MIN_TEXT_BLOCK_CHARS       = 40
REPEATED_TEXT_THRESHOLD    = 3
REPEATED_TEXT_MAX_CHARS    = 200

# Legacy summarization concurrency for optional paths.
TEXT_SUMMARY_MAX_CONCURRENCY  = 4
TABLE_SUMMARY_MAX_CONCURRENCY = 4
IMAGE_SUMMARY_MAX_CONCURRENCY = 3

# ------------------------------------------------------------------
# Chroma Vector Store  (disk-persistent)
# ------------------------------------------------------------------
CHROMA_PERSIST_DIR     = os.path.normpath(
    os.path.join(_PROJECT_ROOT, "chroma_db")
)
CHROMA_COLLECTION_NAME = "mm_rag"

# ------------------------------------------------------------------
# Data directories
# ------------------------------------------------------------------
PDF_DATA_DIR       = os.path.normpath(os.path.join(_PROJECT_ROOT, "data"))
PDF_DATA_DIR2      = os.path.normpath(os.path.join(_PROJECT_ROOT, "data2"))
EXTRACTED_DIR      = os.path.normpath(os.path.join(_PROJECT_ROOT, "extracted_data"))
EXTRACTED_DIR2     = os.path.normpath(os.path.join(_PROJECT_ROOT, "extracted_data2"))
DEBUG_IMAGE_DIR    = os.path.normpath(os.path.join(_PROJECT_ROOT, "debug_images"))

# Ensure output directories exist
for _d in (EXTRACTED_DIR, EXTRACTED_DIR2, DEBUG_IMAGE_DIR):
    os.makedirs(_d, exist_ok=True)

print("pdf_config loaded: Tesseract, Poppler, Gemini, and Chroma config set successfully!")
