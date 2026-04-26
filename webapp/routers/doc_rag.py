# -*- coding: utf-8 -*-
"""
webapp/routers/doc_rag.py
--------------------------
FastAPI router for the Document RAG pipeline (PDF processing + querying).

Endpoints:
    POST /api/doc/process   — upload a PDF and run the full pipeline
    GET  /api/doc/stream/{job_id} — SSE log stream for a running job
    POST /api/doc/query     — query the active RAG chain
    POST /api/doc/reset     — wipe extracted images + chroma_db
    GET  /api/doc/status    — collection stats
"""

import os
import sys
import uuid
import json
import shutil
import queue
import threading
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent.parent.resolve()
SRC_DIR       = PROJECT_ROOT / "Document  Multimodal Summrizer"
UPLOADS_DIR   = PROJECT_ROOT / "webapp" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ────────────────────────────────────────────
_jobs: dict = {}          # job_id → {"q": Queue, "status": str}

# ── Pipeline state (persists across requests within one server session) ──
_chain     = None
_retriever = None


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

class _StdoutCapture:
    """Thread-local stdout redirector that writes to a queue."""
    def __init__(self, q: queue.Queue):
        self.q = q
    def write(self, msg: str):
        if msg and msg.strip():
            self.q.put(("log", msg.strip()))
    def flush(self):
        pass
    def isatty(self):
        return False


def _run_pipeline_thread(job_id: str, pdf_path: str, img_dir: str, force: bool):
    """Run the document RAG pipeline in a background thread."""
    global _chain, _retriever
    q = _jobs[job_id]["q"]
    old_stdout = sys.stdout
    sys.stdout = _StdoutCapture(q)
    try:
        # Lazy-import so sys.path is already set by app.py
        import pdf_config as config  # noqa
        from pdf_pipeline import run_pipeline
        from pdf_retriever import build_vectorstore, create_multi_vector_retriever
        from pdf_rag_chain import build_rag_chain

        # run_pipeline builds retriever and chain internally — we rebuild
        # them here to keep references in module state for /query
        from extract_text   import extract_text_from_raw, summarize_texts
        from extract_tables import extract_tables, summarize_tables
        from extract_images import extract_images, generate_img_summaries
        from unstructured.partition.pdf import partition_pdf
        import shutil, tempfile

        os.makedirs(img_dir, exist_ok=True)
        existing_jpg = sum(1 for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))

        if not force and existing_jpg > 0:
            print(f"Found {existing_jpg} existing images in '{os.path.basename(img_dir)}' — skipping re-extraction")
            tmp = tempfile.mkdtemp(prefix="mmrag_skip_")
            try:
                raw = partition_pdf(
                    filename=pdf_path, strategy="hi_res",
                    extract_image_in_pdf=True,
                    extract_image_block_types=["Image", "Table"],
                    extract_image_block_to_payload=False,
                    extract_image_block_output_dir=tmp,
                    poppler_path=config.POPPLER_PATH,
                )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"Partitioning: {os.path.basename(pdf_path)} …")
            raw = partition_pdf(
                filename=pdf_path, strategy="hi_res",
                extract_image_in_pdf=True,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=False,
                extract_image_block_output_dir=img_dir,
                poppler_path=config.POPPLER_PATH,
            )

        print(f"Total raw elements: {len(raw)}")

        _, _, _, narrative, text, _ = extract_text_from_raw(raw)
        tables  = extract_tables(raw)
        all_txt = text + narrative

        print("Summarising text …")
        txt_sums = summarize_texts(all_txt) if all_txt else []
        print("Summarising tables …")
        tab_sums = summarize_tables(tables) if tables else []
        print("Summarising images …")
        imgs_b64, img_sums = generate_img_summaries(img_dir)

        print("Building vector store …")
        vs = build_vectorstore()
        _retriever = create_multi_vector_retriever(vs, txt_sums, all_txt, tab_sums, tables, img_sums, imgs_b64)
        _chain = build_rag_chain(_retriever)
        print("Pipeline complete. RAG chain ready ✓")
        q.put(("done", None))
    except Exception as exc:
        q.put(("error", str(exc)))
    finally:
        sys.stdout = old_stdout
        _jobs[job_id]["status"] = "done"


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@router.post("/process")
async def process_pdf(
    file: UploadFile = File(...),
    force_reset: bool = Form(default=False),
):
    """Upload a PDF and start the RAG pipeline. Returns a job_id for SSE streaming."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file
    pdf_path = str(UPLOADS_DIR / file.filename)
    content  = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)

    # Image extraction directory (next to the PDF but outside uploads)
    safe_stem = Path(file.filename).stem
    img_dir   = str(PROJECT_ROOT / f"extracted_{safe_stem}")

    if force_reset:
        # Wipe extracted images and chroma DB
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir, ignore_errors=True)
        chroma_dir = str(PROJECT_ROOT / "chroma_db")
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir, ignore_errors=True)
        global _chain, _retriever
        _chain, _retriever = None, None

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"q": queue.Queue(), "status": "running"}

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, pdf_path, img_dir, force_reset),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "filename": file.filename}


@router.get("/stream/{job_id}")
async def stream_logs(job_id: str):
    """SSE endpoint: stream pipeline logs for the given job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def generator():
        q = _jobs[job_id]["q"]
        while True:
            try:
                msg_type, msg_data = q.get_nowait()
                if msg_type == "log":
                    yield f"data: {json.dumps({'type': 'log', 'msg': msg_data})}\n\n"
                elif msg_type == "done":
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                elif msg_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'msg': msg_data})}\n\n"
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                await asyncio.sleep(0.3)

    return StreamingResponse(generator(), media_type="text/event-stream")


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
def query_doc(req: QueryRequest):
    """Query the active document RAG chain."""
    if _chain is None:
        raise HTTPException(status_code=400, detail="No document indexed yet. Process a PDF first.")
    from pdf_rag_chain import query as rag_query
    answer = rag_query(_chain, req.question)
    return {"answer": answer, "question": req.question}


@router.post("/reset")
def reset_doc():
    """Wipe all extracted images and the Chroma vector store."""
    global _chain, _retriever
    removed = []

    for d in (PROJECT_ROOT / "extracted_data", PROJECT_ROOT / "extracted_data2"):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))

    # Also remove any extracted_<stem> dirs from uploads
    for d in PROJECT_ROOT.iterdir():
        if d.is_dir() and d.name.startswith("extracted_"):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))

    chroma = PROJECT_ROOT / "chroma_db"
    if chroma.exists():
        shutil.rmtree(chroma, ignore_errors=True)
        removed.append(str(chroma))

    _chain, _retriever = None, None
    return {"reset": True, "removed": removed}


@router.get("/status")
def doc_status():
    """Return Chroma collection size and extracted image count."""
    try:
        import pdf_config as config
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        col    = client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
        vec_count = col.count()
    except Exception:
        vec_count = -1

    img_count = 0
    for d in PROJECT_ROOT.iterdir():
        if d.is_dir() and d.name.startswith("extracted"):
            img_count += sum(1 for f in d.iterdir() if f.suffix.lower() == ".jpg")

    return {
        "chain_ready": _chain is not None,
        "chroma_vectors": vec_count,
        "extracted_images": img_count,
    }
