# -*- coding: utf-8 -*-
"""
pdf_extractor.py
----------------
Partition a PDF document once using Unstructured and return raw elements.

This is the shared partition step — all other PDF RAG modules (extract_text,
extract_tables, extract_images) receive raw_pdf_elements from here so the
expensive PDF partition call only happens once per pipeline run.

Functions
---------
partition_pdf_once(pdf_path, img_output_dir, force_reextract) -> list
    Partition a PDF and return raw Unstructured elements.
    Skips re-extracting images if they already exist on disk.
"""

import os
import shutil
import tempfile
from pdf2image import convert_from_path

from unstructured.partition.pdf import partition_pdf

import pdf_config as config


def _existing_jpg_count(directory: str) -> int:
    """Return the number of .jpg files already in *directory*."""
    if not os.path.exists(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.lower().endswith(".jpg"))


def _manual_extract_images(pdf_path: str, output_dir: str):
    """
    Manually extract each page of the PDF as an image to bypass unstructured hi_res bugs.
    """
    print(f"  [Manual Image Extraction] Extracting images to {os.path.basename(output_dir)} ...")
    try:
        images = convert_from_path(pdf_path, poppler_path=config.POPPLER_PATH)
        for i, img in enumerate(images, start=1):
            img_path = os.path.join(output_dir, f"figure-{i}.jpg")
            img.save(img_path, "JPEG")
        print(f"  Extracted {len(images)} images.")
    except Exception as e:
        print(f"  [Error] Manual image extraction failed: {e}")


def partition_pdf_once(
    pdf_path: str,
    img_output_dir: str,
    force_reextract: bool = False,
) -> list:
    """
    Partition a PDF a single time and return raw Unstructured elements.

    If ``force_reextract=False`` (default) and .jpg images already exist in
    ``img_output_dir``, image extraction is skipped.

    Parameters
    ----------
    pdf_path        : Absolute path to the PDF file.
    img_output_dir  : Directory where extracted image/table files are written.
    force_reextract : Set True to force fresh image extraction on every run.

    Returns
    -------
    list  — Raw Unstructured elements (text, tables, images).
    """
    os.makedirs(img_output_dir, exist_ok=True)
    existing = _existing_jpg_count(img_output_dir)

    if not force_reextract and existing > 0:
        print(f"  Found {existing} existing image(s) in '{os.path.basename(img_output_dir)}'")
        print("  Skipping image re-extraction (pass force_reextract=True to override).")
        return partition_pdf(
            filename=pdf_path,
            strategy="ocr_only",
            poppler_path=config.POPPLER_PATH,
        )

    print(f"  Partitioning PDF: {os.path.basename(pdf_path)} ...")
    elements = partition_pdf(
        filename=pdf_path,
        strategy="ocr_only",
        poppler_path=config.POPPLER_PATH,
    )
    
    # Manually extract images since ocr_only doesn't do it
    _manual_extract_images(pdf_path, img_output_dir)
    
    return elements


# ------------------------------------------------------------------
# Quick smoke-test (run: python pdf_extractor.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import pdf_config  # noqa: F401
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True, help="PDF path")
    args = parser.parse_args()

    OUTPUT_DIR = config.EXTRACTED_DIR2

    raw = partition_pdf_once(args.pdf, OUTPUT_DIR)
    print(f"[pdf_extractor] Total elements extracted: {len(raw)}")
    print(f"[pdf_extractor] Element types: {set(type(e).__name__ for e in raw)}")
