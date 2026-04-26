# -*- coding: utf-8 -*-
"""
extract_images.py
-----------------
Functions and methods to extract, encode, and summarize IMAGES from PDF documents.

Requirements (pip install):
    unstructured[all-docs]
    pillow
    langchain_core
    langchain-google-genai

Usage:
    from extract_images import extract_images, encode_image, image_summarize, generate_img_summaries
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import base64  # Binary-to-text encoding for image payloads
import os

import config  # load Tesseract, Poppler, and API key config

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------------------------------------------------------
# Image Extraction from raw PDF elements
# ------------------------------------------------------------------

def extract_images(raw_pdf_elements):
    """
    Filter and collect Image elements from partitioned PDF elements.

    Args:
        raw_pdf_elements (list): Output from partition_pdf().

    Returns:
        list[str]: String representations of all Image elements.
    """
    img = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Image" in str(type(element)):
            img.append(str(element))
    return img


# ------------------------------------------------------------------
# Image Encoding
# ------------------------------------------------------------------

def encode_image(image_path):
    """
    Encode an image file to a Base64 string.

    Args:
        image_path (str): Path to the image file (.jpg, .png, etc.)

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ------------------------------------------------------------------
# Image Summarization
# ------------------------------------------------------------------

def image_summarize(img_base64, prompt):
    """
    Generate a summary for a single Base64-encoded image using Gemini Vision.

    Args:
        img_base64 (str): Base64-encoded image string.
        prompt     (str): Instruction prompt for the model.

    Returns:
        str: Model-generated summary of the image.
    """
    chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=1024)

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

def generate_img_summaries(path):
    """
    Generate summaries and Base64 strings for all .jpg images in a folder.

    Args:
        path (str): Directory containing .jpg images extracted by Unstructured.

    Returns:
        tuple: (img_base64_list, image_summaries)
            - img_base64_list (list[str]): Base64 strings of each image.
            - image_summaries (list[str]): Model-generated summaries.
    """
    img_base64_list  = []
    image_summaries  = []

    prompt = (
        "You are an assistant tasked with summarizing images for retrieval. "
        "These summaries will be embedded and used to retrieve the raw image. "
        "Give a concise summary of the image that is well optimized for retrieval."
    )

    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path     = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries


# ------------------------------------------------------------------
# Main (standalone usage example)
# ------------------------------------------------------------------

if __name__ == "__main__":
    EXTRACTED_IMAGE_DIR = "extracted_data2/"

    img_base64_list, image_summaries = generate_img_summaries(EXTRACTED_IMAGE_DIR)

    print(f"Total images processed: {len(img_base64_list)}")
    if image_summaries:
        print("\nFirst image summary:")
        print(image_summaries[0])
