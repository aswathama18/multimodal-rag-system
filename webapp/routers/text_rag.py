# -*- coding: utf-8 -*-
"""
webapp/routers/text_rag.py
---------------------------
FastAPI router for the pure Text RAG pipeline.

Endoints:
    POST /api/text/process  — upload a txt file or paste text -> build FAISS index
    POST /api/text/query    — query the text RAG chain
    POST /api/text/reset    — clear the FAISS index
    GET  /api/text/status   — return chain readiness
"""

import os
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ── Pipeline state (persists across requests within one server session) ──
_chain = None
_vectorstore = None

@router.post("/process")
async def process_text(
    file: Optional[UploadFile] = File(None),
    pasted_text: Optional[str] = Form(None),
):
    """Process a text file or pasted text to build a FAISS vector store."""
    if not file and not pasted_text:
        raise HTTPException(status_code=400, detail="Must provide either a file or pasted text.")
    
    global _chain, _vectorstore
    
    # Extract text content
    text_content = ""
    if pasted_text:
        text_content = pasted_text
    elif file:
        if not file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt files are supported for this endpoint.")
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
        from model_loader import load_text_model

        docs = get_text_chunks(text_content)
        embeddings = build_embeddings()
        _vectorstore = build_vectorstore(docs, embeddings)
        retriever = build_retriever(_vectorstore)
        llm = load_text_model()
        
        _chain = build_text_rag_chain(retriever, llm)
        
        return {
            "status": "success", 
            "message": f"Successfully indexed {len(docs)} text chunks.",
            "chunk_count": len(docs)
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error building Text RAG pipeline: {str(exc)}")

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
def query_text(req: QueryRequest):
    """Query the active Text RAG chain."""
    if _chain is None:
        raise HTTPException(status_code=400, detail="No text indexed yet. Process text first.")
    try:
        answer = _chain.invoke(req.question)
        return {"answer": answer, "question": req.question}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query error: {str(exc)}")

@router.post("/reset")
def reset_text():
    """Wipe the active text chain and vector store."""
    global _chain, _vectorstore
    _chain, _vectorstore = None, None
    return {"reset": True, "message": "Text RAG pipeline reset."}

@router.get("/status")
def text_status():
    """Return status of the Text RAG pipeline."""
    count = 0
    if _vectorstore is not None:
        count = _vectorstore.index.ntotal if hasattr(_vectorstore, 'index') else -1
        
    return {
        "chain_ready": _chain is not None,
        "indexed_chunks": count
    }
