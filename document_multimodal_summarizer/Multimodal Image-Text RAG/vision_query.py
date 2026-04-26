# -*- coding: utf-8 -*-
"""
vision_query.py
---------------
Helpers for building and sending vision queries to the Gemini vision model.

Functions
---------
build_image_message(prompt, image) -> HumanMessage
    Construct a LangChain HumanMessage containing both a text prompt and
    a PIL Image (for the vision model).

query_vision_model(vision_model, prompt, image) -> str
    Send a single image + prompt to the vision model and return the text response.

Source: Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio_Yt (1).ipynb
  - HumanMessage with image_url (cells: zJ116KMkqSfU, 3-kXDQdVtaMZ)
  - vision_model.invoke([message]) (cells: UaIyYPpPqN8h, exYSFX8Vtkym)
"""

from PIL import Image
from langchain_core.messages import HumanMessage


# ------------------------------------------------------------------
# Message builder
# ------------------------------------------------------------------

def build_image_message(prompt: str, image: Image.Image) -> HumanMessage:
    """
    Build a LangChain HumanMessage containing a text prompt and a PIL image.

    This format is compatible with ChatGoogleGenerativeAI (gemini-pro-vision).

    Parameters
    ----------
    prompt : The text instruction / question for the vision model.
    image  : PIL.Image.Image — the image to analyse.

    Returns
    -------
    HumanMessage  — ready to pass to vision_model.invoke([message]).
    """
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": image,       # PIL Image accepted by LangChain Gemini adapter
            },
        ]
    )
    return message


# ------------------------------------------------------------------
# One-shot vision query
# ------------------------------------------------------------------

def query_vision_model(vision_model, prompt: str, image: Image.Image) -> str:
    """
    Send a single image + prompt to a Gemini vision model and return the response.

    Parameters
    ----------
    vision_model : ChatGoogleGenerativeAI vision model instance.
    prompt       : The text instruction / question for the vision model.
    image        : PIL.Image.Image — the image to analyse.

    Returns
    -------
    str — The model's text response.
    """
    message = build_image_message(prompt, image)
    response = vision_model.invoke([message])
    return response.content


# ------------------------------------------------------------------
# Quick smoke-test (run: python vision_query.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import image_text_config  # noqa: F401
    from model_loader import load_vision_model
    from image_handler import get_image

    DEMO_URL = (
        "https://static.nike.com/a/images/t_PDP_1728_v1/"
        "f_auto,q_auto:eco/1705ca64-fbc8-4b79-a451-4ab77760c219/dunk-low-older-shoes-C7T1cx.png"
    )
    img = get_image(DEMO_URL, "nike-shoes-vision-test", "png")
    vision = load_vision_model()
    answer = query_vision_model(
        vision,
        prompt="Give me a summary of this image in 5 words.",
        image=img,
    )
    print(f"[vision_query] Vision model response: {answer}")
