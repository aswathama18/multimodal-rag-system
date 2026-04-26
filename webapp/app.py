# -*- coding: utf-8 -*-
"""
webapp/app.py
-------------
FastAPI entry point for the Multimodal RAG Dashboard.

Run from the project root:
    uvicorn webapp.app:app --reload --port 8000
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Resolve directories ────────────────────────────────────────────
WEBAPP_DIR     = Path(__file__).parent.resolve()
PROJECT_ROOT   = WEBAPP_DIR.parent
SRC_DIR        = PROJECT_ROOT / "Document  Multimodal Summrizer"
VIDEO_DIR      = SRC_DIR / "Video RAG"
PDF_DIR        = SRC_DIR / "PDF RAG"
IMAGE_TEXT_DIR = SRC_DIR / "Multimodal Image-Text RAG"

# Inject source dirs into sys.path so routers can import pipeline modules
for p in (str(SRC_DIR), str(VIDEO_DIR), str(PDF_DIR), str(IMAGE_TEXT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal RAG Dashboard",
    description="Local web interface for Document RAG and Video RAG",
    version="1.0.0",
)

# ── Include routers ────────────────────────────────────────────────
from webapp.routers import doc_rag, video_rag, image_analyzer, image_text_rag, text_rag   # noqa: E402

app.include_router(doc_rag.router,  prefix="/api/doc",   tags=["Document RAG"])
app.include_router(video_rag.router, prefix="/api/video", tags=["Video RAG"])
app.include_router(image_analyzer.router, prefix="/api/image", tags=["Image Analyzer"])
app.include_router(image_text_rag.router, prefix="/api/imagetext", tags=["Image-Text RAG"])
app.include_router(text_rag.router, prefix="/api/text", tags=["Text RAG"])

# ── Static files & SPA root ───────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")

@app.get("/")
def index():
    return FileResponse(str(WEBAPP_DIR / "static" / "index.html"))

@app.get("/api/status")
def overall_status():
    """Quick health-check endpoint."""
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "src_dir": str(SRC_DIR),
    }
