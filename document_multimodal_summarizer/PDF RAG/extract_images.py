# -*- coding: utf-8 -*-
"""
extract_images.py
-----------------
Extract, encode, and summarise IMAGE elements from partitioned PDF raw elements.

Functions
---------
extract_images(raw_pdf_elements) -> list[str]
    Filter Image elements from Unstructured raw elements.

encode_image(image_path) -> str
    Base64-encode a local image file.

image_summarize(img_base64, prompt) -> str
    Generate a summary for a single Base64-encoded image via Gemini Vision.

generate_img_summaries(path) -> tuple[list, list]
    Batch-generate summaries and Base64 strings for all .jpg images in a folder.

Source: Document Multimodal Summrizer/extract_images.py  (ported to PDF RAG folder)
"""

import base64
import os

import pdf_config as config

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# ------------------------------------------------------------------
# Image Element Extraction (from raw PDF elements)
# ------------------------------------------------------------------

def extract_images(raw_pdf_elements: list) -> list:
    """
    Filter and collect Image elements from partitioned PDF elements.

    Parameters
    ----------
    raw_pdf_elements : Output from ``pdf_extractor.partition_pdf_once()``.

    Returns
    -------
    list[str]  — String representations of all Image elements.
    """
    img = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Image" in str(type(element)):
            img.append(str(element))
    print(f"[extract_images] Found {len(img)} image reference(s) in raw elements.")
    return img


# ------------------------------------------------------------------
# Base64 Encoding
# ------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    """
    Encode an image file to a Base64 string.

    Parameters
    ----------
    image_path : Path to the image file (.jpg, .png, etc.).

    Returns
    -------
    str  — Base64-encoded image bytes as UTF-8 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ------------------------------------------------------------------
# Single-image Summarisation
# ------------------------------------------------------------------

def image_summarize(img_base64: str, prompt: str) -> str:
    """
    Generate a summary for a single Base64-encoded image using Gemini Vision.

    Uses GEMINI_FLASH_MODEL (fast vision model) defined in pdf_config.py.

    Parameters
    ----------
    img_base64 : Base64-encoded image string.
    prompt     : Instruction prompt for the vision model.

    Returns
    -------
    str  — Model-generated summary of the image.
    """
    chat = ChatGoogleGenerativeAI(model=config.GEMINI_FLASH_MODEL, max_tokens=1024)

    msg = chat.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ]
        )
    ])
    return msg.content


# ------------------------------------------------------------------
# Batch Image Summary Generation
# ------------------------------------------------------------------

def generate_img_summaries(path: str) -> tuple:
    """
    Generate summaries and Base64 strings for all .jpg images in a folder.

    Reads every .jpg file in ``path`` (sorted), encodes it to Base64, and
    generates a retrieval-optimised summary with the Gemini Vision model.

    Parameters
    ----------
    path : Directory containing .jpg images extracted by Unstructured.

    Returns
    -------
    tuple  — (img_base64_list, image_summaries)
        img_base64_list : list[str]  — Base64 string for each image.
        image_summaries : list[str]  — Model-generated summary for each image.
    """
    img_base64_list = []
    image_summaries = []

    summarize_prompt = (
        "You are an assistant tasked with summarizing images for retrieval. "
        "These summaries will be embedded and used to retrieve the raw image. "
        "Give a concise summary of the image that is well optimized for retrieval."
    )

    if not os.path.exists(path):
        print(f"[extract_images] WARNING: image directory not found — {path}")
        return img_base64_list, image_summaries

    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path     = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, summarize_prompt))

    print(f"[extract_images] Generated {len(image_summaries)} image summary(ies).")
    return img_base64_list, image_summaries


# ------------------------------------------------------------------
# Quick smoke-test (run: python extract_images.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import pdf_config  # noqa: F401

    EXTRACTED_IMAGE_DIR = config.EXTRACTED_DIR2

    img_base64_list, image_summaries = generate_img_summaries(EXTRACTED_IMAGE_DIR)

    print(f"\nTotal images processed: {len(img_base64_list)}")
    if image_summaries:
        print(f"\nFirst image summary:\n{image_summaries[0]}")
