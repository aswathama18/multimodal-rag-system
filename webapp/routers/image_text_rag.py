# -*- coding: utf-8 -*-
"""
webapp/routers/image_text_rag.py
--------------------------------
FastAPI router for the Image-Text RAG pipeline.

Endpoints:
    POST /api/imagetext/process  — upload txt file/paste text -> build FAISS index
    POST /api/imagetext/query    — query the chain (with optional image)
    POST /api/imagetext/reset    — wipe the FAISS index
    GET  /api/imagetext/status   — return chain readiness
"""

import os
import io
from typing import Optional
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ── Pipeline state (persists across requests within one server session) ──
_rag_chain = None
_vision_model = None
_vectorstore = None

@router.post("/process")
async def process_text_kb(
    file: Optional[UploadFile] = File(None),
    pasted_text: Optional[str] = Form(None),
):
    """Process the text knowledge base (file or pasted) to build the FAISS vector store."""
    if not file and not pasted_text:
        raise HTTPException(status_code=400, detail="Must provide either a text file or pasted text.")
    
    global _rag_chain, _vision_model, _vectorstore
    
    text_content = ""
    if pasted_text:
        text_content = pasted_text
    elif file:
        if not file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt files are supported for the knowledge base.")
        content = await file.read()
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content.decode("latin-1")

    if not text_content.strip():
        raise HTTPException(status_code=400, detail="Text content is empty.")

    try:
        import image_text_config  # noqa
        from text_processor import get_text_chunks, build_embeddings, build_vectorstore, build_retriever
        from rag_chain import build_text_rag_chain
        from model_loader import load_text_model, load_vision_model

        docs = get_text_chunks(text_content)
        embeddings = build_embeddings()
        _vectorstore = build_vectorstore(docs, embeddings)
        retriever = build_retriever(_vectorstore)
        
        text_llm = load_text_model()
        _vision_model = load_vision_model()
        
        _rag_chain = build_text_rag_chain(retriever, text_llm)
        
        return {
            "status": "success", 
            "message": f"Successfully indexed {len(docs)} text chunks.",
            "chunk_count": len(docs)
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error building Image-Text RAG pipeline: {str(exc)}")

@router.post("/query")
async def query_image_text(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    """Query the RAG chain, optionally with an image."""
    if _rag_chain is None:
        raise HTTPException(status_code=400, detail="No knowledge base indexed yet. Process text first.")
        
    try:
        from rag_chain import build_full_multimodal_chain
        from langchain_core.messages import HumanMessage
        
        if image:
            # Full multimodal query
            content = await image.read()
            pil_img = Image.open(io.BytesIO(content))
            
            # Use the vision model to describe the image and pass that description to the RAG chain
            full_chain = build_full_multimodal_chain(_rag_chain, _vision_model)
            
            message = HumanMessage(content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": pil_img},
            ])
            
            answer = full_chain.invoke([message])
            return {"answer": answer, "question": question, "image_included": True}
        else:
            # Standard text query
            answer = _rag_chain.invoke(question)
            return {"answer": answer, "question": question, "image_included": False}
            
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query error: {str(exc)}")

@router.post("/reset")
def reset_image_text():
    """Wipe the active chain and vector store."""
    global _rag_chain, _vision_model, _vectorstore
    _rag_chain, _vision_model, _vectorstore = None, None, None
    return {"reset": True, "message": "Image-Text RAG pipeline reset."}

@router.get("/status")
def image_text_status():
    """Return status of the Image-Text RAG pipeline."""
    count = 0
    if _vectorstore is not None:
        count = _vectorstore.index.ntotal if hasattr(_vectorstore, 'index') else -1
        
    return {
        "chain_ready": _rag_chain is not None,
        "indexed_chunks": count
    }
