# ============================================================
#  MULTIMODAL RAG — Issues, Notes & Pre-Deployment Checklist
#  Last updated: 2026-04-22
# ============================================================

## STATUS KEY
#  [DONE]   — Already fixed in current codebase
#  [TODO]   — Known issue to fix before deployment
#  [LATER]  — Planned for upscale / future sprint
#  [NOTE]   — Informational, no action required yet


# ════════════════════════════════════════════════════════════
#  SECTION 1 — CODE & PIPELINE ISSUES
# ════════════════════════════════════════════════════════════

[DONE]  Retriever stray top-level calls
        → retriever.py had plt_img_base64(img_base64_list[1]) and other
          live calls outside any function / guard, crashing on import.
          Fixed: all stray calls removed.

[DONE]  Invalid Gemini model names in API calls
        → "Gemini 2.5 Pro" (capital G + spaces) and "gemini-3.1-flash-lite"
          (model doesn't exist) would cause API 404 errors at runtime.
          Fixed: centralised in config.py as GEMINI_PRO_MODEL and
          GEMINI_FLASH_LITE_MODEL with correct API identifiers.

[DONE]  ChatOpenAI / OpenAI dependency in rag_chain
        → multi_modal_rag_chain() used ChatOpenAI("gpt-4-vision-preview")
          which requires a paid OpenAI API key and is inconsistent with
          the rest of the project using Google Gemini.
          Fixed: replaced with ChatGoogleGenerativeAI(GEMINI_PRO_MODEL).

[DONE]  extracted_data2/ overflow on every run
        → partition_pdf wrote new image files every pipeline run,
          accumulating duplicates indefinitely.
          Fixed: _partition_once() checks for existing JPG files and
          redirects image output to a discarded temp dir if they exist.
          Use force_reextract=True to override.

[DONE]  Chroma vector store was in-memory only
        → All embeddings were lost on every process exit.
          Fixed: Chroma now persists to disk at ../chroma_db/
          (project root level, shareable across future apps).

[DONE]  Double PDF partition calls
        → Each extractor (text, tables, images) was calling partition_pdf
          independently = 3x slow PDF processing per run.
          Fixed: pipeline.py calls partition_pdf once and passes raw
          elements to all three extractors via extract_text_from_raw().

[TODO]  Chroma docstore is still InMemoryStore
        → The vector index (embeddings) persists to disk, but the
          docstore that maps doc_id to raw content (text/table/image)
          is still InMemoryStore — rebuilt from scratch every run.
          On restart, retriever.invoke() will find vectors but return
          empty content.
        FIX NEEDED: Replace InMemoryStore with a SQLite or Redis-backed
          store before deployment. Options:
          - langchain_community.storage.SQLiteStore (lightweight, no deps)
          - Redis (better for multi-instance deployment)

[TODO]  No error handling around Gemini API calls
        → summarize_texts(), summarize_tables(), image_summarize() have
          no retry logic or rate-limit handling.
          If Gemini returns a 429 (quota exceeded) or 500, the whole
          pipeline crashes mid-batch.
        FIX NEEDED: Wrap batch calls with tenacity retry decorator
          or try/except with exponential backoff.

[TODO]  generate_img_summaries reads ALL jpg files in the directory
        → If extracted_data2/ contains leftover images from a different
          PDF or a partial failed run, they will all be embedded.
        FIX NEEDED: Track which jpg files were written in the current
          partition and only load those.

[TODO]  Text and NarrativeText lists can be very large
        → For long PDFs, summarize_texts(all_texts) with max_concurrency=2
          will be extremely slow. No chunking or length limit.
        FIX NEEDED: Add a max_chars guard per element before summarisation
          and consider chunking long NarrativeText blocks.

[TODO]  No deduplication of Chroma entries across runs
        → On the second pipeline run, all summaries are re-embedded and
          added to Chroma again, creating duplicate vectors.
        FIX NEEDED: Check if a collection already has documents before
          calling add_documents(), or use a hash-based document ID.


# ════════════════════════════════════════════════════════════
#  SECTION 2 — SECURITY ISSUES (CRITICAL FOR DEPLOYMENT)
# ════════════════════════════════════════════════════════════

[DONE]  GOOGLE_API_KEY is hardcoded in config.py and notebooks
        → API key is visible in plain text in source code.
          This is a critical security risk — anyone with access to the
          repo or codebase can use your API quota.
        Fixed: 
          1. Removed the key from config.py and multimodal_rag2_0_using_llamaindex.py
          2. Now loading with: os.environ.get("GOOGLE_API_KEY")

[TODO]  No .gitignore present
        → chroma_db/, debug_images/, extracted_data/, .env would all be
          committed to Git by default.
        FIX NEEDED: Create .gitignore before any Git init or push.

[TODO]  Absolute file paths hardcoded (Windows-specific)
        → PDF_PATH, EXTRACTED_IMG_DIR, poppler_path, tesseract_bin all
          use hardcoded C:\Users\nikhi\... paths.
          This will break on any other machine or deployment server.
        FIX NEEDED: Move all paths to .env or a config YAML file
          loaded at startup.


# ════════════════════════════════════════════════════════════
#  SECTION 3 — DEPLOYMENT / SCALING READINESS
# ════════════════════════════════════════════════════════════

[IN PROGRESS]  No interactive query interface for end users
        → Currently only pipeline.py with hardcoded EXAMPLE_QUERIES.
          query() and batch_query() exist in rag_chain.py but require
          the caller to pass question strings programmatically.
        BUILDING: FastAPI + Vanilla HTML/CSS/JS local web app
          → webapp/app.py + routers/ + static/
          → 5 tabs: Document RAG, Image Analyzer, Video RAG, Text RAG, Image-Text RAG
          → Force-Reset buttons clear all data stacks for respective pipelines
          → Run: uvicorn webapp.app:app --reload --port 8000

[LATER] No multi-PDF / multi-document support
        → Pipeline currently processes one PDF at a time.
          For a general-purpose app, users need to upload and query
          across multiple documents.
        PLANNED: Add a document registry / metadata store so Chroma
          collections can be scoped per user or per document set.

[LATER] No user authentication / session management
        → For public deployment, each user needs an isolated context
          (different retriever, different query history).
        PLANNED: Integrate with an auth layer before exposing to public.

[LATER] No logging / observability
        → print() statements are the only output — nothing is persisted
          for debugging production failures.
        PLANNED: Replace prints with Python logging module and route
          to a log file. Add request tracing for the API layer.

[LATER] Chroma not suitable for multi-instance deployment
        → Chroma's local disk mode (sqlite) doesn't support concurrent
          writes from multiple server processes.
        PLANNED: Switch to Chroma server mode or a managed vector DB
          (e.g. Pinecone, Weaviate, Vertex AI Vector Search) before
          scaling beyond a single server.

[NOTE]  Tesseract and Poppler must be installed as system binaries
        → Not Python packages — must be installed separately on any
          machine or deployment container.
          Documented in install_requirements.txt.
          For Docker deployment: include in Dockerfile as apt-get installs.

[NOTE]  gemini-2.5-pro-preview-05-06 is a preview model
        → Preview models can be deprecated or renamed by Google with
          short notice. Monitor the Google AI changelog.
          Stable fallback: "gemini-1.5-pro"


# ════════════════════════════════════════════════════════════
#  SECTION 4 — QUICK REFERENCE: FILE STRUCTURE
# ════════════════════════════════════════════════════════════

Multi_modal_RAG/
│
├── .env                   ← [NEW] API key + system paths (gitignored)
├── .gitignore             ← [NEW] covers .env, chroma_db, lancedb, etc.
│
├── Document  Multimodal Summrizer/        <- source modules
│   ├── config.py          paths (from .env), model names, Chroma config
│   ├── extract_text.py    text extraction + summarisation (Gemini Flash Lite)
│   ├── extract_tables.py  table extraction + summarisation (Gemini Pro)
│   ├── extract_images.py  image extraction + summarisation (Gemini Flash)
│   ├── retriever.py       Chroma vectorstore + MultiVectorRetriever builder
│   ├── display.py         render helpers + terminal image fallback
│   ├── rag_chain.py       Gemini RAG chain + query() + batch_query()
│   ├── pipeline.py        end-to-end entry point
│   └── Video RAG/
│       ├── video_config.py     [UPDATED] loads from .env, no hardcoded URL
│       ├── video_processor.py  [FIXED] overflow guard on download + frames
│       ├── video_indexer.py    [FIXED] lancedb existence check
│       ├── video_retriever.py  retrieval + matplotlib display
│       ├── video_llm.py        Gemini multimodal QA
│       └── main_video.py       CLI entry point
│
├── webapp/                ← [NEW] FastAPI local web app
│   ├── app.py             main FastAPI app (mounts static, routers)
│   ├── routers/
│   │   ├── doc_rag.py         /api/doc/* — PDF pipeline
│   │   ├── image_analyzer.py  /api/image/* — Gemini Vision
│   │   ├── video_rag.py       /api/video/* — Video pipeline
│   │   ├── text_rag.py        /api/text/* — Pure Text RAG
│   │   └── image_text_rag.py  /api/imagetext/* — Combined Image-Text RAG
│   └── static/
│       ├── index.html     5-tab SPA
│       ├── style.css      premium dark glass UI with Markdown support
│       └── app.js         tab logic + fetch + live log stream + markdown rendering
│
├── chroma_db/             Chroma on-disk vector index (auto-created)
├── debug_images/          terminal image renders (auto-created)
├── extracted_data/        images from PDF 1 (Cji.pdf)
├── extracted_data2/       images from PDF 2 (RAG NLP paper)
├── data/                  source PDFs (set 1)
├── data2/                 source PDFs (set 2)
├── requirements.txt       pip install -r requirements.txt
└── install_requirements.txt  Tesseract + Poppler system install notes


# ════════════════════════════════════════════════════════════
#  SECTION 5 — VIDEO RAG ISSUES
# ════════════════════════════════════════════════════════════

[DONE]  video_processor.py — download_video() overflow
        → Re-downloaded input_vid.mp4 every run even if it existed.
          Fixed: checks os.path.exists(video_file); skips if found.
          Use force=True to re-download.

[DONE]  video_processor.py — video_to_images() overflow
        → Re-extracted all frames every run, accumulating duplicates
          in mixed_data/ (41 → 82 → 123... frames per run).
          Fixed: checks glob("frame*.png"); skips if frames found.
          Use force=True to re-extract.

[DONE]  video_indexer.py — LanceDB duplicate indexing
        → from_documents() was called every run, re-embedding all
          documents and creating duplicate vectors in LanceDB.
          Fixed: _lancedb_has_data() checks table row count;
          reconnects to existing index if data is present.
          Use force_reindex=True to rebuild.

[DONE]  video_config.py — hardcoded VIDEO_URL
        → URL was hardcoded: VIDEO_URL = "https://youtu.be/..."
          Fixed: VIDEO_URL removed from config. The web app and
          CLI callers must provide the URL as a function argument.

[DONE]  video_config.py + config.py — API key hardcoded
        → GOOGLE_API_KEY was hardcoded in both config files.
          Fixed: both now use python-dotenv to load from .env.
          New API key rotated and stored only in .env.

[DONE]  main_video.py still references video_config.VIDEO_URL
        → main_video.py line 28 passes video_config.VIDEO_URL which
          no longer exists in video_config. The webapp bypasses
          main_video.py and calls pipeline functions directly.
        Fixed: main_video.py now uses argparse to accept --url and --query.
        (Same for pdf_pipeline.py, pdf_extractor.py, and main_image_text.py)


# ════════════════════════════════════════════════════════════
#  SECTION 6 — LOCAL WEB APP ARCHITECTURE
# ════════════════════════════════════════════════════════════

[NOTE]  FastAPI + Vanilla HTML/CSS/JS (no Node/npm required)
        → Run: uvicorn webapp.app:app --reload --port 8000
        → Open: http://localhost:8000

[NOTE]  Five input tabs
        Tab 1 — Document RAG : upload .pdf → full Doc pipeline → query box
        Tab 2 — Image Analyzer : upload image(s) → Gemini Vision direct
        Tab 3 — Video RAG : user pastes YouTube URL → pipeline → query
        Tab 4 — Text RAG : upload .txt → text FAISS index → query
        Tab 5 — Image-Text RAG : upload KB .txt + upload Image → vision → RAG

[NOTE]  Force-Reset behavior
        → Document Reset: clears extracted_data*/ + chroma_db/
        → Video Reset   : clears mixed_data/ + video_data/ + lancedb/
        → Next run after reset is a full fresh first-time run.

[NOTE]  New packages required
        fastapi, uvicorn[standard], python-multipart, python-dotenv
        Added to requirements.txt. Install before running webapp.
