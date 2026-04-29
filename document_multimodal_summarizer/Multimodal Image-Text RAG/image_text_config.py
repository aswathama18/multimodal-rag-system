# -*- coding: utf-8 -*-
"""
image_text_config.py
--------------------
Shared configuration for the Multimodal Image-Text RAG pipeline.

This module sets up:
  - Google Gemini API key (from .env at project root)
  - Model names for text, vision, and embedding
  - FAISS vector store settings
  - Image cache directory

Source: Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio_Yt (1).ipynb
"""

import os
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load .env from project root  (three levels up from this file)
# Folder layout:
#   Multi_modal_RAG/
#     Document Multimodal Summrizer/
#       Multimodal Image-Text RAG/    <-- this file lives here
# ------------------------------------------------------------------
_THIS_FILE    = os.path.abspath(__file__)
_MODULE_DIR   = os.path.dirname(_THIS_FILE)          # Multimodal Image-Text RAG/
_PARENT_DIR   = os.path.dirname(_MODULE_DIR)          # Document Multimodal Summrizer/
_PROJECT_ROOT = os.path.normpath(os.path.join(_PARENT_DIR, ".."))  # Multi_modal_RAG/

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ------------------------------------------------------------------
# Google Gemini API Key  (never hardcode — loaded from .env)
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
# Text-only RAG model (answers from retrieved text context)
GEMINI_TEXT_MODEL   = "gemini-2.5-flash"

# Vision model (analyses images and generates descriptions)
GEMINI_VISION_MODEL = "gemini-2.5-flash"

# Embedding model used by FAISS for semantic search
EMBED_MODEL         = "gemini-embedding-001"

# ------------------------------------------------------------------
# Text Splitter defaults
# ------------------------------------------------------------------
TEXT_CHUNK_SIZE    = 20    # characters  (small — matches notebook demo)
TEXT_CHUNK_OVERLAP = 10    # characters

# ------------------------------------------------------------------
# Image cache directory
# Fetched / downloaded images are stored here during a session.
# ------------------------------------------------------------------
IMAGE_CACHE_DIR = os.path.join(_MODULE_DIR, "image_cache")
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# FAISS index persist path
# Optional: save/load the FAISS index to avoid re-embedding on restart.
# ------------------------------------------------------------------
FAISS_INDEX_DIR = os.path.join(_MODULE_DIR, "faiss_index")
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

print("image_text_config loaded: Gemini models and FAISS config set successfully!")
