# -*- coding: utf-8 -*-
"""
pdf_pipeline.py
---------------
End-to-end entry point for the PDF Multimodal RAG pipeline.

Run from the PDF RAG folder:
    python pdf_pipeline.py

Stages
------
1. Partition the PDF once  (shared across all extractors)
2. Extract text, table, and image elements
3. Summarise each element type with Gemini
4. Build a disk-persistent Chroma MultiVector retriever
5. Build the multimodal RAG chain
6. Run example queries and display retrieved context + answers

Edit the USER CONFIG block below to point at your own PDFs.

Source: Document Multimodal Summrizer/pipeline.py  (ported to PDF RAG folder)
"""

import os
import sys

# Ensure the PDF RAG folder is on sys.path when run directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ── 0. Config MUST be first ──────────────────────────────────────
import pdf_config as config

# ── 1. PDF partition helper ──────────────────────────────────────
from pdf_extractor import partition_pdf_once

# ── 2. Extraction helpers ────────────────────────────────────────
from extract_text   import extract_text_from_raw, extract_ordered_text_blocks, build_text_chunks
from extract_tables import extract_tables, summarize_tables
from extract_images import extract_images, generate_img_summaries

# ── 3. Retriever ─────────────────────────────────────────────────
from pdf_retriever import build_vectorstore, create_multi_vector_retriever

# ── 4. RAG chain ─────────────────────────────────────────────────
from pdf_rag_chain import build_rag_chain, query, batch_query

# ── 5. Display ───────────────────────────────────────────────────
from pdf_display import display_retrieval_results


# ══════════════════════════════════════════════════════════════════
#  USER CONFIG (Defaults)
# ══════════════════════════════════════════════════════════════════

DEFAULT_PDF_PATH  = config.PDF_DATA_DIR2 + "\\sample.pdf"
EXTRACTED_IMG_DIR = config.EXTRACTED_DIR2


# ══════════════════════════════════════════════════════════════════
#  Pipeline runner
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    pdf_path:          str  = DEFAULT_PDF_PATH,
    extracted_img_dir: str  = EXTRACTED_IMG_DIR,
    example_queries:   list = None,
    force_reextract:   bool = False,
    enable_table_summaries: bool = False,
    enable_image_summaries: bool = False,
) -> None:
    """
    Run the full PDF Multimodal RAG pipeline end-to-end.

    Parameters
    ----------
    pdf_path          : Path to the source PDF.
    extracted_img_dir : Directory for extracted images/tables.
    example_queries   : List of questions to run after building the retriever.
    force_reextract   : True → force fresh image extraction even if files exist.
    enable_table_summaries : Summarize tables for retrieval (LLM/API cost).
    enable_image_summaries : Summarize images for retrieval (LLM/API cost).
    """
    print("\n" + "=" * 70)
    print("  PDF MULTIMODAL RAG PIPELINE")
    print("=" * 70)

    # -- Stage 1: Partition ----------------------------------------
    print("\n[Stage 1] Partitioning PDF (one shared pass) ...")
    raw_elements = partition_pdf_once(pdf_path, extracted_img_dir, force_reextract)
    print(f"  Total raw elements: {len(raw_elements)}")

    # -- Stage 2: Extract ------------------------------------------
    print("\n[Stage 2] Extracting element types ...")
    Header, Footer, Title, NarrativeText, Text, ListItem = extract_text_from_raw(raw_elements)
    Tab  = extract_tables(raw_elements)
    _img = extract_images(raw_elements)      # string repr -- for counting only

    print(f"  Text blocks   : {len(Text)}")
    print(f"  NarrativeText : {len(NarrativeText)}")
    print(f"  Tables        : {len(Tab)}")
    print(f"  Image refs    : {len(_img)}")

    # -- Stage 3: Prepare retrieval content ------------------------
    print("\n[Stage 3] Preparing retrieval content ...")
    ordered_text_blocks = extract_ordered_text_blocks(raw_elements)
    text_chunks = build_text_chunks(ordered_text_blocks)

    table_summaries = []
    if enable_table_summaries and Tab:
        try:
            table_summaries = summarize_tables(Tab)
        except Exception as exc:
            print(f"  [Stage 3] WARNING: table summarization skipped due to error: {exc}")
    elif Tab:
        print("  [Stage 3] Skipping table summarization (fast mode).")

    img_base64_list, image_summaries = [], []
    if enable_image_summaries:
        try:
            img_base64_list, image_summaries = generate_img_summaries(extracted_img_dir)
        except Exception as exc:
            print(f"  [Stage 3] WARNING: image summarization skipped due to error: {exc}")
            img_base64_list, image_summaries = [], []
    else:
        print("  [Stage 3] Skipping image summarization (fast mode).")

    print(f"  Text chunks    : {len(text_chunks)}")
    print(f"  Table summaries: {len(table_summaries)}")
    print(f"  Image summaries: {len(image_summaries)}")

    # -- Stage 4: Build retriever ----------------------------------
    print("\n[Stage 4] Building disk-persistent MultiVector retriever ...")
    print(f"  Chroma dir: {config.CHROMA_PERSIST_DIR}")
    vectorstore = build_vectorstore()
    retriever   = create_multi_vector_retriever(
        vectorstore,
        text_chunks, text_chunks,
        table_summaries, Tab,
        image_summaries, img_base64_list,
    )
    print("  Retriever ready v")

    # -- Stage 5: Build RAG chain ----------------------------------
    print("\n[Stage 5] Building RAG chain ...")
    chain = build_rag_chain(retriever)
    print("  Chain ready v")

    # -- Stage 6: Run example queries ------------------------------
    if not example_queries:
        print("\n[Stage 6] No example queries configured -- skipping.")
        return

    print(f"\n[Stage 6] Running {len(example_queries)} example quer(y/ies) ...")

    for q in example_queries:
        print(f"\n{'-'*70}")
        docs = retriever.invoke(q)
        print(f"[Retriever] {len(docs)} document(s) for: {q[:65]}...")
        display_retrieval_results(docs)
        answer = query(chain, q)
        print(f"\n[RAG Answer]\n{answer}")

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print(f"  Debug images saved to: {config.DEBUG_IMAGE_DIR}")
    print("=" * 70 + "\n")


# ==================================================================
#  Entry point
# ==================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PDF Multimodal RAG Pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of images")
    parser.add_argument("--with-tables", action="store_true", help="Enable table summarization for retrieval")
    parser.add_argument("--with-images", action="store_true", help="Enable image summarization for retrieval")
    args = parser.parse_args()
    
    run_pipeline(
        pdf_path=args.pdf,
        example_queries=[args.query],
        force_reextract=args.force,
        enable_table_summaries=args.with_tables,
        enable_image_summaries=args.with_images,
    )
