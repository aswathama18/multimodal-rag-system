# -*- coding: utf-8 -*-
"""
webapp/routers/image_analyzer.py
---------------------------------
Direct Gemini Vision analysis — no indexing required.

Endpoints:
    POST /api/image/analyze — upload 1-3 images + optional prompt → Gemini answer
"""

import os
import base64
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

router = APIRouter()


@router.post("/analyze")
async def analyze_images(
    files: List[UploadFile] = File(...),
    prompt: str = Form(default="Describe this image in detail. What do you see? Provide a thorough analysis."),
):
    """
    Send uploaded image(s) to Gemini Vision and return the analysis.
    Supports JPEG, PNG, WEBP. Max 3 files per request.
    """
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images per request.")

    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    parts   = [prompt]

    for f in files:
        mime = f.content_type or "image/jpeg"
        if mime not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported type: {mime}. Use JPEG, PNG, or WEBP.")
        data = await f.read()
        parts.append({"mime_type": mime, "data": data})

    try:
        import google.generativeai as genai
        model    = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(parts)
        return {"analysis": response.text, "image_count": len(files)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gemini Vision error: {exc}")
