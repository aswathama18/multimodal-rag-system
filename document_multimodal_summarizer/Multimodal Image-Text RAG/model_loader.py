# -*- coding: utf-8 -*-
"""
model_loader.py
---------------
Centralised LLM / vision-model factory for the Multimodal Image-Text RAG pipeline.

Functions
---------
load_text_model()   -> ChatGoogleGenerativeAI  (text-only Gemini)
load_vision_model() -> ChatGoogleGenerativeAI  (vision Gemini)

Source: Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio_Yt (1).ipynb
  - `load_model()` function (cells: v6dATsOFo0VJ, 67oGZQvHo7tC, sZVlrxYLqsNI)
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from image_text_config import GEMINI_TEXT_MODEL, GEMINI_VISION_MODEL


def load_text_model() -> ChatGoogleGenerativeAI:
    """
    Return a ChatGoogleGenerativeAI instance configured for text-only tasks.

    Uses GEMINI_TEXT_MODEL defined in image_text_config.py.
    Suitable for:
        - General Q&A
        - RAG answer generation from retrieved text context

    Returns
    -------
    ChatGoogleGenerativeAI
    """
    return ChatGoogleGenerativeAI(model=GEMINI_TEXT_MODEL)


def load_vision_model() -> ChatGoogleGenerativeAI:
    """
    Return a ChatGoogleGenerativeAI instance configured for multimodal (vision) tasks.

    Uses GEMINI_VISION_MODEL defined in image_text_config.py.
    Suitable for:
        - Describing / summarising images
        - Answering questions about image content

    Returns
    -------
    ChatGoogleGenerativeAI
    """
    return ChatGoogleGenerativeAI(model=GEMINI_VISION_MODEL)


# ------------------------------------------------------------------
# Quick smoke-test (run: python model_loader.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import image_text_config  # noqa: F401 — ensures key is set before first use

    text_model = load_text_model()
    reply = text_model.invoke("Say hello in one sentence.").content
    print(f"[Text model] {reply}")

    vision_model = load_vision_model()
    print(f"[Vision model] loaded: {GEMINI_VISION_MODEL}")
