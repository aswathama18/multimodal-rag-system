# -*- coding: utf-8 -*-
"""
main_image_text.py
------------------
Entry point for the adaptive Multimodal Image-Text RAG pipeline.

Routes:
  1. text_query only             -> FAISS retriever -> Text RAG -> LLM
  2. image + mm_query only       -> Gemini vision -> Final LLM
  3. text_query + image + query  -> FAISS + vision -> merged context -> Final LLM
"""

import os
import sys

from PIL import Image

# Make sure the package directory is on sys.path when running directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Config must be imported before model/embedding construction.
import image_text_config  # noqa: F401

from image_handler import get_image
from model_loader import load_text_model, load_vision_model
from rag_chain import (
    build_full_multimodal_chain,
    build_image_only_chain,
    build_text_rag_chain,
)
from text_processor import (
    build_embeddings,
    build_retriever,
    build_vectorstore,
    get_text_chunks,
    load_text_file,
)


def _clean(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _load_image(image_url=None, image_path=None):
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        image.load()
        return image

    if image_url:
        return get_image(image_url, "product-image", "png")

    return None


def _build_text_components(text_file, llm_text):
    """
    Build FAISS/retriever/text-RAG only for routes that use text retrieval.
    Image-only routing must not call this function.
    """
    if not text_file:
        raise ValueError("--text_file is required for text RAG and hybrid RAG routes.")
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text knowledge base not found: {text_file}")

    print(f"\n[2/5] Loading text from: {text_file}")
    raw_text = load_text_file(text_file)
    docs = get_text_chunks(raw_text)

    print("\n[3/5] Building FAISS vectorstore...")
    embeddings = build_embeddings()
    vectorstore = build_vectorstore(docs, embeddings)
    retriever = build_retriever(vectorstore)
    rag_chain = build_text_rag_chain(retriever, llm_text)
    return retriever, rag_chain


def _detect_route(text_query, image):
    """
    Route by available modalities.

    mm_query is intentionally excluded from text routing. It describes what
    the vision model should reason about and is never sent to the retriever.
    """
    has_text = bool(text_query)
    has_image = image is not None

    if has_image and not has_text:
        return "image_only"
    if has_text and not has_image:
        return "text_only"
    if has_text and has_image:
        return "hybrid"
    raise ValueError("Provide --text_query, --image_url/--image_path with --mm_query, or both.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Multimodal Image-Text RAG")
    parser.add_argument("--text_file", type=str, help="Path to text knowledge base file")
    parser.add_argument("--image_url", type=str, help="URL of image to analyze")
    parser.add_argument("--image_path", type=str, help="Local image path to analyze")
    parser.add_argument("--text_query", type=str, help="Text retrieval query")
    parser.add_argument("--mm_query", type=str, help="Image reasoning query for the vision model")
    args = parser.parse_args()

    text_query = _clean(args.text_query)
    mm_query = _clean(args.mm_query)

    print("\n[1/5] Loading Gemini models...")
    llm_text = load_text_model()
    llm_vision = load_vision_model()

    image = _load_image(args.image_url, args.image_path)
    route = _detect_route(text_query, image)
    print(f"\n[route] Selected pipeline: {route}")

    if route == "image_only":
        print("\n[2/5] Running image-only pipeline; skipping FAISS retriever.")
        image_chain = build_image_only_chain(llm_vision, llm_text)
        answer = image_chain.invoke(
            {
                "image": image,
                "mm_query": mm_query or "Answer the user's question about this image.",
            }
        )
        print(f"  Image Query: {mm_query or 'Describe the image.'}")
        print(f"  Answer: {answer}")

    elif route == "text_only":
        retriever, rag_chain = _build_text_components(args.text_file, llm_text)
        print("\n[4/5] Running text-only RAG query...")
        answer = rag_chain.invoke(text_query)
        print(f"  Query : {text_query}")
        print(f"  Answer: {answer}")

    else:
        retriever, _ = _build_text_components(args.text_file, llm_text)
        print("\n[4/5] Running hybrid RAG query...")
        full_chain = build_full_multimodal_chain(retriever, llm_vision, llm_text)
        answer = full_chain.invoke(
            {
                "text_query": text_query,
                "mm_query": mm_query or "Describe the image with details relevant to the text query.",
                "image": image,
            }
        )
        print(f"  Text Query : {text_query}")
        print(f"  Image Query: {mm_query or 'Describe the image with details relevant to the text query.'}")
        print(f"  Answer: {answer}")

    print("\nMultimodal Image-Text RAG pipeline completed successfully.")
