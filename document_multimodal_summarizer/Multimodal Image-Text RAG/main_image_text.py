# -*- coding: utf-8 -*-
"""
main_image_text.py
------------------
Entry point for the Multimodal Image-Text RAG pipeline.

This script reproduces the full notebook workflow:
  1. Load a text knowledge base (e.g. nike_shoes.txt).
  2. Build a FAISS vectorstore + retriever from the text.
  3. Compose a text-only RAG chain.
  4. Compose the full multimodal chain (vision → RAG).
  5. Download a product image.
  6. Run a pure-text RAG query (no image needed).
  7. Run the full multimodal query (image → vision → RAG → final answer).

Usage
-----
    python main_image_text.py

Source: Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio_Yt (1).ipynb
  (All cells combined into a single runnable script)
"""

import os
import sys

# Make sure the package directory is on sys.path when running directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ------------------------------------------------------------------
# 0. Config  (must be first — sets GOOGLE_API_KEY env var)
# ------------------------------------------------------------------
import image_text_config  # noqa: F401

# ------------------------------------------------------------------
# 1. Load Gemini Models
# ------------------------------------------------------------------
from model_loader import load_text_model, load_vision_model

print("\n[1/7] Loading Gemini models...")
llm_text   = load_text_model()
llm_vision = load_vision_model()

print(" Gemini text model loaded successfully .")

# Quick sanity-check on text model
# reply = llm_text.invoke("please come up with the best funny line.").content
# print(f"  Text model test: {reply}")

# ------------------------------------------------------------------
# 2. Load and chunk the knowledge-base text
# ------------------------------------------------------------------
from text_processor import (
    load_text_file,
    get_text_chunks,
    build_embeddings,
    build_vectorstore,
    build_retriever,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal Image-Text RAG")
    parser.add_argument("--text_file", type=str, required=True, help="Path to text knowledge base file")
    parser.add_argument("--image_url", type=str, help="URL of image to analyze")
    parser.add_argument("--text_query", type=str, required=True, help="Text query to ask")
    parser.add_argument("--mm_query", type=str, help="Multimodal query to ask (requires image)")
    args = parser.parse_args()

    TEXT_FILE_PATH = args.text_file

    print(f"\n[2/7] Loading text from: {TEXT_FILE_PATH}")
    if not os.path.exists(TEXT_FILE_PATH):
        print(f"  WARNING: {TEXT_FILE_PATH} not found. Exiting.")
        sys.exit(1)
    
    raw_text = load_text_file(TEXT_FILE_PATH)
    docs = get_text_chunks(raw_text)

    # ------------------------------------------------------------------
    # 3. Build FAISS vectorstore & retriever
    # ------------------------------------------------------------------
    print("\n[3/7] Building FAISS vectorstore...")
    embeddings  = build_embeddings()
    vectorstore = build_vectorstore(docs, embeddings)
    retriever   = build_retriever(vectorstore)

    # Test retriever
    test_results = retriever.invoke("Nike slide/sandal.")
    print(f"  Retriever test results: {[r.page_content for r in test_results]}")

    # ------------------------------------------------------------------
    # 4. Build RAG chains
    # ------------------------------------------------------------------
    from rag_chain import build_text_rag_chain, build_full_multimodal_chain

    print("\n[4/7] Building RAG chains...")
    rag_chain  = build_text_rag_chain(retriever, llm_text)
    full_chain = build_full_multimodal_chain(rag_chain, retriever, llm_vision)

    # ------------------------------------------------------------------
    # 5. Download a product image for the multimodal demo
    # ------------------------------------------------------------------
    from image_handler import get_image, display_image

    print("\n[5/7] Downloading demo product image...")
    product_image = None
    if args.image_url:
        try:
            product_image = get_image(args.image_url, "product-image", "png")
            display_image(product_image, title="Product Image")
        except Exception as e:
            print(f"  WARNING: Could not download image — {e}")

    # ------------------------------------------------------------------
    # 6. Text-only RAG query
    # ------------------------------------------------------------------
    print("\n[6/7] Text-only RAG query...")
    text_query  = args.text_query
    text_answer = rag_chain.invoke(text_query)
    print(f"  Query : {text_query}")
    print(f"  Answer: {text_answer}")

    # ------------------------------------------------------------------
    # 7. Full multimodal query (image → vision → RAG)
    # ------------------------------------------------------------------
    if product_image is not None and args.mm_query:
        from vision_query import build_image_message

        print("\n[7/7] Full multimodal query (image + text → vision → RAG)...")
        mm_prompt = args.mm_query
        message   = build_image_message(mm_prompt, product_image)
        inputs = {
            "text_query": args.mm_query or args.text_query,
            "image": product_image 
        }

        mm_answer = full_chain.invoke(inputs)
        print("  Multimodal Query : ", mm_prompt)
        print(f"  Answer: {mm_answer}")
    else:
        print("\n[7/7] Skipped multimodal query (no image or mm_query provided).")

    print("\n✅  Multimodal Image-Text RAG pipeline completed successfully.")
