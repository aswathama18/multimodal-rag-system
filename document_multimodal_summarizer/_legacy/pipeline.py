# -*- coding: utf-8 -*-
"""
pipeline.py
-----------
End-to-end entry point for the Multimodal RAG pipeline.

Run from the project's source directory:
    python pipeline.py

Stages:
    1. Partition the PDF once (shared across all extractors)
    2. Extract text, table, and image elements
    3. Summarise each element type with Gemini
    4. Build a disk-persistent Chroma MultiVector retriever
    5. Build the multimodal RAG chain
    6. Run example queries and display retrieved context + answers

Edit the ``USER CONFIG`` block below to point to your own PDFs.
"""

import os

# ── 0. Config MUST be first — sets API key, Tesseract, Poppler ───
import config

# ── 1. Extraction helpers ─────────────────────────────────────────
from unstructured.partition.pdf import partition_pdf  # shared partition

from extract_text   import extract_text_from_raw, summarize_texts
from extract_tables import extract_tables, summarize_tables
from extract_images import extract_images, generate_img_summaries

# ── 2. Retriever ──────────────────────────────────────────────────
from retriever import build_vectorstore, create_multi_vector_retriever

# ── 3. RAG chain ──────────────────────────────────────────────────
from rag_chain import build_rag_chain, query, batch_query

# ── 4. Display ────────────────────────────────────────────────────
from display import display_retrieval_results


# ══════════════════════════════════════════════════════════════════
#  USER CONFIG — edit these paths for your own PDFs
# ══════════════════════════════════════════════════════════════════

PDF_PATH          = r"c:\Users\nikhi\OneDrive\Desktop\Multi_modal_RAG\data2\Retrieval-Augmented Generation for NLP.pdf"
EXTRACTED_IMG_DIR = r"c:\Users\nikhi\OneDrive\Desktop\Multi_modal_RAG\extracted_data2"

# Sample queries to run after building the retriever
EXAMPLE_QUERIES = [
    "Why do we combine a pre-trained retriever with a pre-trained seq2seq model (Generator) and fine-tune end-to-end?",
    "Explain the NQ performance graph as more documents are retrieved.",
    "What are the Open-Domain QA test scores described in the paper?",
]


# ══════════════════════════════════════════════════════════════════
#  Pipeline stages
# ══════════════════════════════════════════════════════════════════

import shutil
import tempfile


def _existing_jpg_count(directory: str) -> int:
    """Return the number of .jpg files already in *directory*."""
    if not os.path.exists(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.lower().endswith(".jpg"))


def _partition_once(
    pdf_path:        str,
    img_output_dir:  str,
    force_reextract: bool = False,
) -> list:
    """
    Partition the PDF a single time and return raw elements.

    If ``force_reextract=False`` (default) and JPG images already exist in
    ``img_output_dir``, the function skips writing images to disk — it routes
    image output to a temporary directory that is discarded immediately.
    This prevents ``extracted_data2/`` from accumulating duplicate files on
    every pipeline run.

    Pass ``force_reextract=True`` when you deliberately want a fresh extraction
    (e.g. after changing the source PDF).

    Args:
        pdf_path        : Absolute path to the PDF.
        img_output_dir  : Directory where extracted image files are written.
        force_reextract : If True, always write images even if they exist.

    Returns:
        list: Raw unstructured elements (text, tables, images).
    """
    os.makedirs(img_output_dir, exist_ok=True)
    existing = _existing_jpg_count(img_output_dir)

    if not force_reextract and existing > 0:
        # ── Images already extracted — partition for text/tables only,
        #    redirect image output to a throwaway temp dir so the real
        #    img_output_dir stays clean.
        print(f"  Found {existing} existing image(s) in '{os.path.basename(img_output_dir)}'")
        print("  Skipping image re-extraction (pass force_reextract=True to override)")
        tmp_dir = tempfile.mkdtemp(prefix="mmrag_skip_")
        try:
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                extract_image_in_pdf=True,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=False,
                extract_image_block_output_dir=tmp_dir,   # ← temp, not the real dir
                poppler_path=config.poppler_path,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)   # ← discard temp images
        return elements

    # ── First run (or forced refresh) — extract normally ─────────────
    print(f"  Partitioning: {os.path.basename(pdf_path)} …")
    return partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_image_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=img_output_dir,
        poppler_path=config.poppler_path,
    )


def run_pipeline(
    pdf_path:          str  = PDF_PATH,
    extracted_img_dir: str  = EXTRACTED_IMG_DIR,
    example_queries:   list = EXAMPLE_QUERIES,
    force_reextract:   bool = False,
) -> None:
    """
    Run the full Multimodal RAG pipeline end-to-end.

    Args:
        pdf_path          : Path to the source PDF.
        extracted_img_dir : Directory for extracted images/tables.
        example_queries   : List of questions to run through the RAG chain.
        force_reextract   : Set True to force fresh image extraction even if
                            files already exist in extracted_img_dir.
    """
    print("\n" + "═" * 70)
    print("  MULTIMODAL RAG PIPELINE")
    print("═" * 70)

    # ── Stage 1: Partition ────────────────────────────────────────
    print("\n[Stage 1] Partitioning PDF (one pass, shared across all extractors) …")
    raw_elements = _partition_once(pdf_path, extracted_img_dir, force_reextract)
    print(f"  Total raw elements: {len(raw_elements)}")


    # ── Stage 2: Extract ──────────────────────────────────────────
    print("\n[Stage 2] Extracting element types …")

    Header, Footer, Title, NarrativeText, Text, ListItem = extract_text_from_raw(raw_elements)
    Tab  = extract_tables(raw_elements)
    _img = extract_images(raw_elements)   # string repr (for counting)

    print(f"  Text blocks   : {len(Text)}")
    print(f"  NarrativeText : {len(NarrativeText)}")
    print(f"  Tables        : {len(Tab)}")
    print(f"  Image refs    : {len(_img)}")

    # ── Stage 3: Summarise ────────────────────────────────────────
    print("\n[Stage 3] Generating summaries with Gemini …")

    # Combine Text + NarrativeText for richer text coverage
    all_texts = Text + NarrativeText

    text_summaries  = summarize_texts(all_texts)   if all_texts else []
    table_summaries = summarize_tables(Tab)         if Tab       else []

    # generate_img_summaries reads .jpg files written during partitioning
    img_base64_list, image_summaries = generate_img_summaries(extracted_img_dir)

    print(f"  Text summaries : {len(text_summaries)}")
    print(f"  Table summaries: {len(table_summaries)}")
    print(f"  Image summaries: {len(image_summaries)}")

    # ── Stage 4: Build retriever ──────────────────────────────────
    print("\n[Stage 4] Building disk-persistent MultiVector retriever …")
    print(f"  Chroma dir: {config.CHROMA_PERSIST_DIR}")

    vectorstore = build_vectorstore()
    retriever   = create_multi_vector_retriever(
        vectorstore,
        text_summaries,  all_texts,
        table_summaries, Tab,
        image_summaries, img_base64_list,
    )
    print("  Retriever ready ✓")

    # ── Stage 5: Build RAG chain ──────────────────────────────────
    print("\n[Stage 5] Building RAG chain …")
    print(f"  Model: {config.GEMINI_PRO_MODEL}")
    chain = build_rag_chain(retriever)
    print("  Chain ready ✓")

    # ── Stage 6: Run example queries ──────────────────────────────
    if not example_queries:
        print("\n[Stage 6] No example queries configured — skipping.")
        return

    print(f"\n[Stage 6] Running {len(example_queries)} example quer(y/ies) …")

    for q in example_queries:
        print(f"\n{'─'*70}")

        # Show what the retriever found (with terminal image fallback)
        docs = retriever.invoke(q)
        print(f"[Retriever] {len(docs)} document(s) for: {q[:65]}…")
        display_retrieval_results(docs)

        # Get the full RAG answer
        answer = query(chain, q)
        print(f"\n[RAG Answer]\n{answer}")

    print("\n" + "═" * 70)
    print("  Pipeline complete.")
    print(f"  Debug images (if any) saved to: {config.DEBUG_IMAGE_DIR}")
    print("═" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline()
