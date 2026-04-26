# -*- coding: utf-8 -*-
"""
webapp/routers/video_rag.py
----------------------------
FastAPI router for the Video RAG pipeline (YouTube download → frame extraction
→ LanceDB indexing → multimodal query).

Endpoints:
    POST /api/video/process        — provide YouTube URL, run pipeline
    GET  /api/video/stream/{job_id}— SSE log stream
    POST /api/video/query          — query the active video index
    POST /api/video/reset          — wipe mixed_data, video_data, lancedb
    GET  /api/video/status         — frame count + lancedb stats
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

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# ── Paths (video_config will set these too, but we need them before import) ──
_VIDEO_SRC = Path(__file__).parent.parent.parent / "Document  Multimodal Summrizer" / "Video RAG"

# ── In-memory job store ────────────────────────────────────────────
_jobs: dict = {}

# ── Pipeline session state ─────────────────────────────────────────
_index            = None
_retriever_engine = None
_llm              = None
_metadata: dict   = {}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

class _StdoutCapture:
    def __init__(self, q: queue.Queue):
        self.q = q
    def write(self, msg: str):
        if msg and msg.strip():
            self.q.put(("log", msg.strip()))
    def flush(self):
        pass
    def isatty(self):
        return False


def _run_video_thread(job_id: str, url: str, force: bool):
    global _index, _retriever_engine, _llm, _metadata
    q = _jobs[job_id]["q"]
    old_stdout = sys.stdout
    sys.stdout = _StdoutCapture(q)
    try:
        import video_config as cfg
        import video_processor
        import video_indexer
        import video_llm

        # Step 1 — Download video
        print("[Step 1] Downloading video …")
        meta = video_processor.download_video(url, cfg.OUTPUT_VIDEO_DIR, force=force)
        _metadata = meta or {}
        if meta:
            print(f"  Title: {meta.get('Title')} | Author: {meta.get('Author')}")

        # Step 2 — Extract frames
        print("[Step 2] Extracting frames …")
        video_processor.video_to_images(cfg.VIDEO_FILE_PATH, cfg.OUTPUT_MIXED_DIR, force=force)

        # Step 3 — Extract + transcribe audio
        print("[Step 3] Extracting & transcribing audio …")
        video_processor.video_to_audio(cfg.VIDEO_FILE_PATH, cfg.OUTPUT_AUDIO_PATH)
        text_data = video_processor.audio_to_text(cfg.OUTPUT_AUDIO_PATH)
        transcript_path = os.path.join(cfg.OUTPUT_MIXED_DIR, "output_text.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text_data)
        if os.path.exists(cfg.OUTPUT_AUDIO_PATH):
            os.remove(cfg.OUTPUT_AUDIO_PATH)
        print("  Transcript saved.")

        # Step 4 — Build / reload index
        print("[Step 4] Building multimodal index …")
        _index = video_indexer.build_video_index(
            cfg.OUTPUT_MIXED_DIR,
            force_reindex=force,
        )
        _retriever_engine = _index.as_retriever(
            similarity_top_k=1, image_similarity_top_k=3
        )

        # Step 5 — Init LLM
        print("[Step 5] Initialising LLM …")
        _llm = video_llm.setup_multimodal_llm()

        print("Video RAG pipeline complete ✓")
        q.put(("done", None))
    except Exception as exc:
        q.put(("error", str(exc)))
    finally:
        sys.stdout = old_stdout
        _jobs[job_id]["status"] = "done"


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    url: str
    force_reset: bool = False


@router.post("/process")
def process_video(req: VideoRequest):
    """Start the video RAG pipeline for the given YouTube URL."""
    url = req.url.strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL.")

    if req.force_reset:
        import video_config as cfg
        global _index, _retriever_engine, _llm, _metadata
        for d in (cfg.OUTPUT_MIXED_DIR, cfg.OUTPUT_VIDEO_DIR, cfg.LANCEDB_URI):
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        _index = _retriever_engine = _llm = None
        _metadata = {}

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"q": queue.Queue(), "status": "running"}

    thread = threading.Thread(
        target=_run_video_thread,
        args=(job_id, url, req.force_reset),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@router.get("/stream/{job_id}")
async def stream_logs(job_id: str):
    """SSE endpoint: stream video pipeline logs."""
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
def query_video(req: QueryRequest):
    """Query the active video RAG index."""
    if _retriever_engine is None or _llm is None:
        raise HTTPException(
            status_code=400,
            detail="No video indexed yet. Process a YouTube URL first."
        )
    import video_retriever
    import video_llm as vllm

    imgs, texts = video_retriever.retrieve_multimodal(_retriever_engine, req.question)
    answer = vllm.answer_query(
        llm=_llm,
        query_str=req.question,
        retrieved_images_paths=imgs,
        retrieved_texts=texts,
        metadata_dict=_metadata,
    )
    return {"answer": answer, "question": req.question, "frames_used": len(imgs)}


@router.post("/reset")
def reset_video():
    """Wipe mixed_data, video_data, and LanceDB tables."""
    global _index, _retriever_engine, _llm, _metadata
    removed = []
    import video_config as cfg
    for d in (cfg.OUTPUT_MIXED_DIR, cfg.OUTPUT_VIDEO_DIR, cfg.LANCEDB_URI):
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    _index = _retriever_engine = _llm = None
    _metadata = {}
    return {"reset": True, "removed": removed}


@router.get("/status")
def video_status():
    """Return frame count, transcript presence, and LanceDB table stats."""
    import video_config as cfg

    frame_count = 0
    has_transcript = False
    if os.path.isdir(cfg.OUTPUT_MIXED_DIR):
        import glob
        frame_count    = len(glob.glob(os.path.join(cfg.OUTPUT_MIXED_DIR, "frame*.png")))
        has_transcript = os.path.exists(os.path.join(cfg.OUTPUT_MIXED_DIR, "output_text.txt"))

    lance_rows = 0
    try:
        import lancedb
        db = lancedb.connect(cfg.LANCEDB_URI)
        if "text_collection" in db.table_names():
            lance_rows = db.open_table("text_collection").count_rows()
    except Exception:
        pass

    return {
        "index_ready":    _retriever_engine is not None,
        "frames":         frame_count,
        "has_transcript": has_transcript,
        "lancedb_rows":   lance_rows,
        "metadata":       _metadata,
    }
