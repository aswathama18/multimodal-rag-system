# -*- coding: utf-8 -*-
"""
extract_text.py
---------------
Functions and methods to extract and summarize TEXT from PDF documents.

Requirements (pip install):
    unstructured[all-docs]
    langchain_core
    langchain
    langchain_community
    langchain-google-genai

Usage:
    from extract_text import extract_text_elements, summarize_texts
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
# Text Extraction
# ------------------------------------------------------------------

def extract_text_from_raw(raw_pdf_elements: list) -> tuple:
    """
    Extract all text-based elements from already-partitioned PDF elements.

    Use this when you have already called ``partition_pdf()`` and want to
    avoid re-partitioning the document.  This is the preferred helper in
    ``pipeline.py`` because it shares a single partition pass.

    Args:
        raw_pdf_elements (list): Output from ``partition_pdf()``.

    Returns:
        tuple: (Header, Footer, Title, NarrativeText, Text, ListItem)
               Each is a list of string representations of the elements.
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

    return Header, Footer, Title, NarrativeText, Text, ListItem


def extract_text_elements(pdf_path: str, output_dir: str) -> tuple:
    """
    Partition a PDF and extract all text-based elements.

    Internally calls ``partition_pdf()`` then delegates to
    ``extract_text_from_raw()``.  Use ``extract_text_from_raw()`` directly
    in ``pipeline.py`` if you are sharing a single partition call.

    Args:
        pdf_path   (str): Absolute path to the PDF file.
        output_dir (str): Directory to store extracted image/table blocks.

    Returns:
        tuple: (Header, Footer, Title, NarrativeText, Text, ListItem)
               Each is a list of string representations of the elements.
    """
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_image_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=output_dir,
        poppler_path=config.poppler_path,
    )
    return extract_text_from_raw(raw_pdf_elements)


# ------------------------------------------------------------------
# Text Summarization
# ------------------------------------------------------------------

def summarize_texts(text_list):
    """
    Generate concise summaries for a list of text strings using Gemini.

    Args:
        text_list (list[str]): List of raw text strings to summarize.

    Returns:
        list[str]: List of generated text summaries.
    """
    prompt_text = (
        "You are an assistant tasked with summarizing text for retrieval. "
        "These summaries will be embedded and used to retrieve the raw text elements. "
        "Give a concise summary of the table or text that is well optimized for retrieval."
        "text: {element}"
    )

    prompt = ChatPromptTemplate.from_template(prompt_text)
    # Uses GEMINI_FLASH_LITE_MODEL ("gemini-2.0-flash-lite") — fast & cheap for bulk jobs
    model  = ChatGoogleGenerativeAI(temperature=0, model=config.GEMINI_FLASH_LITE_MODEL)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = summarize_chain.batch(text_list, {"max_concurrency": 2})
    return text_summaries


# ------------------------------------------------------------------
# Main (standalone usage example)
# ------------------------------------------------------------------

if __name__ == "__main__":
    PDF_PATH   = r"c:\Users\nikhi\OneDrive\Desktop\Multi_modal_RAG\data2\Retrieval-Augmented Generation for NLP.pdf"
    OUTPUT_DIR = r"c:\Users\nikhi\OneDrive\Desktop\Multi_modal_RAG\extracted_data2"

    Header, Footer, Title, NarrativeText, Text, ListItem = extract_text_elements(PDF_PATH, OUTPUT_DIR)

    print(f"Headers      : {len(Header)}")
    print(f"Footers      : {len(Footer)}")
    print(f"Titles       : {len(Title)}")
    print(f"NarrativeText: {len(NarrativeText)}")
    print(f"Text blocks  : {len(Text)}")
    print(f"ListItems    : {len(ListItem)}")

    if Text:
        text_summaries = summarize_texts(Text)
        print("\nFirst text summary:")
        print(text_summaries[0])
