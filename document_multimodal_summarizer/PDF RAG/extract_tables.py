# -*- coding: utf-8 -*-
"""
extract_tables.py
-----------------
Extract and summarise TABLE elements from partitioned PDF raw elements.

Functions
---------
extract_tables(raw_pdf_elements) -> list[str]
    Filter Table elements from Unstructured raw elements.

summarize_tables(tab_list) -> list[str]
    Generate concise retrieval-optimised summaries using Gemini Pro.

Source: Document Multimodal Summrizer/extract_tables.py  (ported to PDF RAG folder)
"""

import pdf_config as config

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# ------------------------------------------------------------------
# Table Extraction
# ------------------------------------------------------------------

def extract_tables(raw_pdf_elements: list) -> list:
    """
    Filter and collect Table elements from partitioned PDF elements.

    Parameters
    ----------
    raw_pdf_elements : Output from ``pdf_extractor.partition_pdf_once()``.

    Returns
    -------
    list[str]  — String representations of all Table elements found.
    """
    Tab = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            Tab.append(str(element))
    print(f"[extract_tables] Found {len(Tab)} table(s).")
    return Tab


# ------------------------------------------------------------------
# Table Summarisation
# ------------------------------------------------------------------

def summarize_tables(tab_list: list) -> list:
    """
    Generate concise summaries for a list of table strings using Gemini.

    Uses GEMINI_PRO_MODEL (best quality) for complex table comprehension.

    Parameters
    ----------
    tab_list : List of raw table strings (HTML or plain text).

    Returns
    -------
    list[str]  — Generated summaries, one per input table.
    """
    if not tab_list:
        return []

    prompt_text = (
        "You are an assistant tasked with summarizing tables for retrieval. "
        "These summaries will be embedded and used to retrieve the raw table elements. "
        "Give a concise summary of the table that is well optimized for retrieval. "
        "Table:{element}"
    )

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model  = ChatGoogleGenerativeAI(model=config.GEMINI_PRO_MODEL)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    summaries = summarize_chain.batch(tab_list, {"max_concurrency": 5})
    print(f"[extract_tables] Generated {len(summaries)} table summaries.")
    return summaries


# ------------------------------------------------------------------
# Quick smoke-test (run: python extract_tables.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import pdf_config  # noqa: F401
    from pdf_extractor import partition_pdf_once

    PDF_PATH   = r"c:\Users\nikhi\OneDrive\Desktop\Multi_modal_RAG\data2\Retrieval-Augmented Generation for NLP.pdf"
    OUTPUT_DIR = config.EXTRACTED_DIR2

    raw = partition_pdf_once(PDF_PATH, OUTPUT_DIR)
    Tab = extract_tables(raw)

    if Tab:
        print(f"\nFirst table (raw):\n{Tab[0]}")
        summaries = summarize_tables(Tab[:1])  # summarise first table for smoke-test
        print(f"\nFirst table summary:\n{summaries[0]}")
