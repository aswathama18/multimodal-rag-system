# -*- coding: utf-8 -*-
"""
image_handler.py
----------------
Utilities for fetching, encoding, and displaying images.

Functions
---------
get_image(url, filename, extension) -> PIL.Image
    Download an image from a URL, save it locally, and return a PIL Image object.

encode_image(image_path) -> str
    Base64-encode a local image file for sending to vision APIs.

display_image(image)
    Show a PIL Image using matplotlib (terminal/script friendly).

Source: Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio_Yt (1).ipynb
  - `get_image()` function (cell: Xa1fVCCBplBR)
  - `encode_image()` / `generate_img_summaries()` pattern (from main.py)
"""

import os
import base64
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from image_text_config import IMAGE_CACHE_DIR


# ------------------------------------------------------------------
# Fetching images from URL
# ------------------------------------------------------------------

def get_image(url: str, filename: str, extension: str) -> Image.Image:
    """
    Download an image from ``url``, save it to IMAGE_CACHE_DIR, and return
    it as a PIL Image.

    Parameters
    ----------
    url       : Full URL of the image to download.
    filename  : Base name (without extension) for the saved file.
    extension : File extension, e.g. ``"png"`` or ``"jpg"``.

    Returns
    -------
    PIL.Image.Image
    """
    save_path = os.path.join(IMAGE_CACHE_DIR, f"{filename}.{extension}")
    content = requests.get(url, timeout=15).content
    with open(save_path, "wb") as f:
        f.write(content)

    image = Image.open(save_path)
    image.load()  # Force load to memory before returning
    print(f"[image_handler] Image saved to: {save_path}")
    return image


# ------------------------------------------------------------------
# Base64 encoding  (required by Gemini vision API)
# ------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    """
    Read a local image file and return its Base64-encoded string.

    Parameters
    ----------
    image_path : Absolute or relative path to an image file.

    Returns
    -------
    str  — Base64-encoded image bytes as UTF-8 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------

def display_image(image: Image.Image, title: str = "Image Preview") -> None:
    """
    Display a PIL Image using matplotlib (works in both Jupyter and terminal).

    Parameters
    ----------
    image : PIL.Image.Image to display.
    title : Window / subplot title string.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def display_image_from_path(image_path: str) -> None:
    """
    Display an image stored on disk using matplotlib.

    Parameters
    ----------
    image_path : Absolute or relative path to an image file.
    """
    img_array = mpimg.imread(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_array)
    plt.axis("off")
    plt.title(os.path.basename(image_path))
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Quick smoke-test (run: python image_handler.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    DEMO_URL  = (
        "https://static.nike.com/a/images/t_PDP_1728_v1/"
        "f_auto,q_auto:eco/1705ca64-fbc8-4b79-a451-4ab77760c219/dunk-low-older-shoes-C7T1cx.png"
    )
    img = get_image(DEMO_URL, "nike-shoes-demo", "png")
    display_image(img, title="Nike Shoes Demo")
    print("[image_handler] Smoke-test passed.")
