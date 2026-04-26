# -*- coding: utf-8 -*-
"""
extract_tables.py
-----------------
Functions and methods to extract and summarize TABLES from PDF documents.

Requirements (pip install):
    unstructured[all-docs]
    lxml
    langchain_core
    langchain
    langchain_community
    langchain-google-genai

Usage:
    from extract_tables import extract_tables, summarize_tables
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import config  # load Tesseract, Poppler, API key, and model names

from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------------------------------------------------------
# Table Extraction from raw PDF elements
# ------------------------------------------------------------------

def extract_tables(raw_pdf_elements):
    """
    Filter and collect Table elements from partitioned PDF elements.

    Args:
        raw_pdf_elements (list): Output from partition_pdf().

    Returns:
        list[str]: String representations of all Table elements.
    """
    Tab = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            Tab.append(str(element))
    return Tab


# ------------------------------------------------------------------
# Table Summarization
# ------------------------------------------------------------------

def summarize_tables(tab_list):
    """
    Generate concise summaries for a list of table strings using Gemini.

    Args:
        tab_list (list[str]): List of raw table strings to summarize.

    Returns:
        list[str]: List of generated table summaries.
    """
    prompt_text = (
        "You are an assistant tasked with summarizing tables for retrieval. "
        "These summaries will be embedded and used to retrieve the raw table elements. "
        "Give a concise summary of the table that is well optimized for retrieval. "
        "Table:{element}"
    )

    prompt = ChatPromptTemplate.from_template(prompt_text)
    # Uses GEMINI_PRO_MODEL ("gemini-2.5-pro-preview-05-06") — best for complex tables
    model  = ChatGoogleGenerativeAI(model=config.GEMINI_PRO_MODEL)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    table_summaries = summarize_chain.batch(tab_list, {"max_concurrency": 5})
    return table_summaries


# ------------------------------------------------------------------
# Main (standalone usage example)
# ------------------------------------------------------------------

if __name__ == "__main__":
    PDF_PATH   = r"c:\Users\nikhi\OneDrive\Desktop\Multi_modal_RAG\data2\Retrieval-Augmented Generation for NLP.pdf"
    OUTPUT_DIR = r"c:\Users\nikhi\OneDrive\Desktop\Multi_modal_RAG\extracted_data2"

    # Partition the PDF to get raw elements
    raw_pdf_elements = partition_pdf(
        filename=PDF_PATH,
        strategy="hi_res",
        extract_image_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=OUTPUT_DIR,
        poppler_path=config.poppler_path
    )

    Tab = extract_tables(raw_pdf_elements)
    print(f"Total tables extracted: {len(Tab)}")

    if Tab:
        print("\nFirst table (raw):")
        print(Tab[0])

        table_summaries = summarize_tables(Tab)
        print("\nFirst table summary:")
        print(table_summaries[0])
