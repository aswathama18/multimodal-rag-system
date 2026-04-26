# -*- coding: utf-8 -*-
"""
video_llm.py
------------
Google Gemini MultiModal LLM setup, QA prompt template, and inference logic.

Requirements (pip install):
    google-generativeai
    nest-asyncio

Usage:
    from video_llm import setup_multimodal_llm, load_image, answer_query
"""

import json
import time

import google.generativeai as genai
import nest_asyncio
from PIL import Image

import video_config  # ensures API keys are set

# Required for running async loops inside some environments (like Colab/Jupyter or async contexts)
nest_asyncio.apply()


def setup_multimodal_llm(model_name="gemini-2.5-flash"):
    """
    Initialize the Gemini multimodal client.

    Args:
        model_name (str): Gemini model to use.

    Returns:
        google.generativeai.GenerativeModel: The configured LLM object.
    """
    genai.configure(api_key=video_config.GOOGLE_API_KEY)
    return genai.GenerativeModel(model_name=model_name)


def load_image(path: str):
    """
    Utility to load an image via PIL for multimodal prompts.
    """
    return Image.open(path)


def answer_query(llm, query_str, retrieved_images_paths, retrieved_texts, metadata_dict):
    """
    Send a prompt with context, metadata, and retrieved images to the LLM to get an answer.

    Args:
        llm: Gemini multimodal client.
        query_str (str): The user's question.
        retrieved_images_paths (list[str]): Paths to retrieved image frames.
        retrieved_texts (list[str]): Retrieved text context.
        metadata_dict (dict): Video metadata.

    Returns:
        str: The LLM's text response.
    """
    qa_tmpl_str = (
        "Based on the provided information, including relevant images and retrieved context from the video, "
        "accurately and precisely answer the query without any additional prior knowledge.\n"
        "-----------------------\n"
        "Context: {context_str}\n"
        "Metadata for video: {metadata_str} \n"
        "-----------------------\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    context_str = "".join(retrieved_texts)
    metadata_str = json.dumps(metadata_dict)
    image_inputs = [load_image(path) for path in retrieved_images_paths if path]

    formatted_prompt = qa_tmpl_str.format(
        query_str=query_str,
        metadata_str=metadata_str,
        context_str=context_str,
    )
    return safe_llm_call(llm, formatted_prompt, image_inputs)


def safe_llm_call(llm, prompt, image_inputs=None, retries=5):
    image_inputs = image_inputs or []
    for attempt in range(retries):
        try:
            result = llm.generate_content([prompt, *image_inputs])
            return result.text
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = 2 ** attempt
                print(f"Model busy, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e
    raise RuntimeError("LLM failed after retries")
