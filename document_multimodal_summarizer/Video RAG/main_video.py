# -*- coding: utf-8 -*-
"""
main_video.py
-------------
Main orchestrator script for the Video RAG pipeline.
Ties together configuration, processing, indexing, retrieving, and LLM synthesis.

Usage:
    python main_video.py
"""

import os
import argparse

# Import modules from our Video RAG package
import video_config
import video_processor
import video_indexer
import video_retriever
import video_llm

def main():
    parser = argparse.ArgumentParser(description="Multimodal Video RAG Pipeline")
    parser.add_argument("--url", type=str, required=True, help="YouTube Video URL")
    parser.add_argument("--query", type=str, required=True, help="Question to ask about the video")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-extract")
    args = parser.parse_args()

    print("=== Multimodal Video RAG Pipeline ===")
    
    # ---------------------------------------------------------
    # 1. Process Video (Download, Extract Frames, Audio, Text)
    # ---------------------------------------------------------
    print("\n[Step 1] Processing Video...")
    print("Clearing cached video, frame, transcript, and LanceDB artifacts...")
    video_processor.clear_video_cache(
        video_config.OUTPUT_VIDEO_DIR,
        video_config.OUTPUT_MIXED_DIR,
        video_config.LANCEDB_URI,
    )

    metadata_vid = video_processor.download_video(args.url, video_config.OUTPUT_VIDEO_DIR, force=args.force)
    
    if metadata_vid:
        print(f"Metadata: {metadata_vid}")
    else:
        print("Warning: Could not get metadata. Proceeding...")
        metadata_vid = {}

    print("Extracting frames...")
    video_processor.video_to_images(
        video_config.VIDEO_FILE_PATH,
        video_config.OUTPUT_MIXED_DIR,
        fps=video_config.FRAME_EXTRACTION_FPS,
        force=args.force,
    )

    print("Extracting audio...")
    video_processor.video_to_audio(video_config.VIDEO_FILE_PATH, video_config.OUTPUT_AUDIO_PATH)

    print("Transcribing audio to text...")
    text_data = video_processor.audio_to_text(video_config.OUTPUT_AUDIO_PATH)
    
    # Save transcript
    transcript_path = os.path.join(video_config.OUTPUT_MIXED_DIR, "output_text.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text_data)
    print("Transcript saved.")

    # Optional cleanup
    if os.path.exists(video_config.OUTPUT_AUDIO_PATH):
        os.remove(video_config.OUTPUT_AUDIO_PATH)
        print("Temporary audio file cleaned up.")


    # ---------------------------------------------------------
    # 2. Build Index
    # ---------------------------------------------------------
    print("\n[Step 2] Building MultiModal Vector Index...")
    index = video_indexer.build_video_index(video_config.OUTPUT_MIXED_DIR, force_reindex=args.force)

    # Create Retriever Engine
    retriever_engine = index.as_retriever(similarity_top_k=1, image_similarity_top_k=3)


    # ---------------------------------------------------------
    # 3. Retrieve and Synthesize Answer
    # ---------------------------------------------------------
    print("\n[Step 3] Querying the Video...")
    
    print(f"Query: {args.query}")

    print("Retrieving context...")
    retrieved_img_paths, retrieved_texts = video_retriever.retrieve_multimodal(retriever_engine, args.query)
    
    print(f"Found {len(retrieved_img_paths)} relevant images and {len(retrieved_texts)} text chunks.")

    print("Initializing LLM...")
    llm = video_llm.setup_multimodal_llm()

    print("Generating Answer...")
    answer = video_llm.answer_query(
        llm=llm,
        query_str=args.query,
        retrieved_images_paths=retrieved_img_paths,
        retrieved_texts=retrieved_texts,
        metadata_dict=metadata_vid
    )

    print("\n================== FINAL ANSWER ==================")
    print(answer)
    print("==================================================")


if __name__ == "__main__":
    main()
