# -*- coding: utf-8 -*-
"""
extract_text.py
---------------
Extract and prepare TEXT elements from partitioned PDF raw elements.

Functions
---------
extract_text_from_raw(raw_pdf_elements) -> tuple
    Filter all text-type elements (Header, Footer, Title, NarrativeText,
    Text, ListItem) from Unstructured raw elements.

extract_ordered_text_blocks(raw_pdf_elements) -> list[str]
    Preserve document-order text blocks for chunk construction.

build_text_chunks(text_blocks) -> list[str]
    Normalize ordered text blocks, merge them into coherent sections, and
    chunk them for direct embedding.

summarize_texts(text_list) -> list[str]
    Legacy helper retained for compatibility. The optimized PDF pipeline
    now embeds raw text chunks directly instead of summarizing each block.

Source: Document Multimodal Summrizer/extract_text.py  (ported to PDF RAG folder)
"""

from collections import Counter
import re

import pdf_config as config

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------------------------------------------------------
# Text Extraction
# ------------------------------------------------------------------

def extract_text_from_raw(raw_pdf_elements: list) -> tuple:
    """
    Extract all text-based elements from already-partitioned PDF elements.

    Prefer this helper inside ``pdf_pipeline.py`` because it reuses
    a single partition call instead of re-partitioning the document.

    Parameters
    ----------
    raw_pdf_elements : Output from ``pdf_extractor.partition_pdf_once()``.

    Returns
    -------
    tuple  — (Header, Footer, Title, NarrativeText, Text, ListItem)
             Each is a list of string representations.
    """
    Header        = []
    Footer        = []
    Title         = []
    NarrativeText = []
    Text          = []
    ListItem      = []

    for element in raw_pdf_elements:
        etype = str(type(element))
        if   "unstructured.documents.elements.Header"        in etype:
            Header.append(str(element))
        elif "unstructured.documents.elements.Footer"        in etype:
            Footer.append(str(element))
        elif "unstructured.documents.elements.Title"         in etype:
            Title.append(str(element))
        elif "unstructured.documents.elements.NarrativeText" in etype:
            NarrativeText.append(str(element))
        elif "unstructured.documents.elements.Text"          in etype:
            Text.append(str(element))
        elif "unstructured.documents.elements.ListItem"      in etype:
            ListItem.append(str(element))

    print(
        f"[extract_text] Found — "
        f"Header:{len(Header)}  Footer:{len(Footer)}  Title:{len(Title)}  "
        f"NarrativeText:{len(NarrativeText)}  Text:{len(Text)}  ListItem:{len(ListItem)}"
    )
    return Header, Footer, Title, NarrativeText, Text, ListItem


def extract_ordered_text_blocks(raw_pdf_elements: list) -> list[str]:
    """Collect retrieval-relevant text blocks while preserving PDF order."""
    accepted_types = (
        "unstructured.documents.elements.Title",
        "unstructured.documents.elements.NarrativeText",
        "unstructured.documents.elements.Text",
        "unstructured.documents.elements.ListItem",
    )
    ordered_blocks: list[str] = []

    for element in raw_pdf_elements:
        etype = str(type(element))
        if any(type_name in etype for type_name in accepted_types):
            ordered_blocks.append(str(element))

    print(f"[extract_text] Collected {len(ordered_blocks)} ordered text block(s) for chunking.")
    return ordered_blocks


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text_block(text: str) -> str:
    """Collapse whitespace and strip trivial punctuation noise."""
    if not text:
        return ""
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.strip("|")


def _filter_boilerplate(blocks: list[str]) -> list[str]:
    """
    Remove empty fragments and repeated boilerplate such as headers/footers.

    Repeated short lines usually harm retrieval more than they help.
    """
    normalized = [_normalize_text_block(block) for block in blocks]
    normalized = [block for block in normalized if len(block) >= config.MIN_TEXT_BLOCK_CHARS]
    counts = Counter(normalized)
    filtered: list[str] = []

    for block in normalized:
        if counts[block] >= config.REPEATED_TEXT_THRESHOLD and len(block) <= config.REPEATED_TEXT_MAX_CHARS:
            continue
        filtered.append(block)

    return filtered


def _merge_text_blocks(blocks: list[str]) -> list[str]:
    """Merge adjacent text blocks into larger sections before chunking."""
    merged: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for block in blocks:
        separator = "\n\n" if current_parts else ""
        projected_length = current_length + len(separator) + len(block)

        if current_parts and projected_length > config.TEXT_SECTION_MAX_CHARS:
            merged.append("\n\n".join(current_parts))
            current_parts = [block]
            current_length = len(block)
            continue

        current_parts.append(block)
        current_length = projected_length

    if current_parts:
        merged.append("\n\n".join(current_parts))

    return merged


def build_text_chunks(text_blocks: list[str]) -> list[str]:
    """
    Build retrieval-ready chunks from raw PDF text elements.

    This is the optimized ingestion path: embed the raw chunks directly
    instead of paying for one LLM call per text element.
    """
    filtered_blocks = _filter_boilerplate(text_blocks)
    merged_sections = _merge_text_blocks(filtered_blocks)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.TEXT_CHUNK_SIZE,
        chunk_overlap=config.TEXT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[str] = []
    for section in merged_sections:
        for chunk in splitter.split_text(section):
            chunk = _normalize_text_block(chunk)
            if len(chunk) >= config.MIN_TEXT_BLOCK_CHARS:
                chunks.append(chunk)

    print(
        f"[extract_text] Built {len(chunks)} retrieval chunk(s) "
        f"from {len(filtered_blocks)} filtered text block(s)."
    )
    return chunks


def summarize_texts(text_list: list) -> list:
    """
    Generate concise summaries for a list of text strings using Gemini.

    Retained for compatibility and small experiments. The optimized PDF
    pipeline no longer uses it during normal ingestion.

    Parameters
    ----------
    text_list : List of raw text strings to summarise.

    Returns
    -------
    list[str]  — Generated summaries, one per input text.
    """
    if not text_list:
        return []

    prompt_text = (
        "You are an assistant tasked with summarizing text for retrieval. "
        "These summaries will be embedded and used to retrieve the raw text elements. "
        "Give a concise summary of the table or text that is well optimized for retrieval."
        "text: {element}"
    )

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model  = ChatGoogleGenerativeAI(temperature=0, model=config.GEMINI_FLASH_MODEL)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    summaries = summarize_chain.batch(
        text_list,
        {"max_concurrency": config.TEXT_SUMMARY_MAX_CONCURRENCY},
    )

    print(f"[extract_text] Generated {len(summaries)} text summaries.")
    return summaries


# ------------------------------------------------------------------
# Quick smoke-test (run: python extract_text.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import pdf_config  # noqa: F401
    from pdf_extractor import partition_pdf_once

    PDF_PATH   = r"c:\Users\nikhi\OneDrive\Desktop\Multi_modal_RAG\data2\Retrieval-Augmented Generation for NLP.pdf"
    OUTPUT_DIR = config.EXTRACTED_DIR2

    raw = partition_pdf_once(PDF_PATH, OUTPUT_DIR)
    Header, Footer, Title, NarrativeText, Text, ListItem = extract_text_from_raw(raw)

    if Text:
        summaries = summarize_texts(Text[:2])  # summarise first 2 for smoke-test
        print(f"\nFirst text summary:\n{summaries[0]}")
