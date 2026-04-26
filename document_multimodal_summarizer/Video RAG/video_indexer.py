# -*- coding: utf-8 -*-
"""
video_indexer.py
----------------
Build (or reload) a LlamaIndex MultiModal vector index from extracted
video frames and transcript stored in a LanceDB vector store.

Overflow Guard:
    build_video_index() checks whether the LanceDB tables already contain
    indexed data. If they do and ``force_reindex=False`` (default), it
    reconnects to the existing index instead of re-embedding every document.
    Pass ``force_reindex=True`` to wipe and rebuild the index from scratch.

Requirements (pip install):
    llama-index
    llama-index-vector-stores-lancedb
    llama-index-embeddings-huggingface
    lancedb

Usage:
    from video_indexer import build_video_index
"""

import os

import video_config  # ensures env vars and paths are set

from llama_index.core                   import SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.indices           import MultiModalVectorStoreIndex
from llama_index.vector_stores.lancedb  import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def _select_index_files(data_folder: str, max_frames: int) -> list[str]:
    """
    Select the transcript plus a capped number of frame images for indexing.

    This keeps first-run multimodal indexing bounded on CPU-only machines.
    """
    entries = sorted(os.listdir(data_folder))
    transcript_files = [
        os.path.join(data_folder, name)
        for name in entries
        if name.lower().endswith(".txt")
    ]
    frame_files = [
        os.path.join(data_folder, name)
        for name in entries
        if name.lower().endswith(".png")
    ]

    if max_frames > 0 and len(frame_files) > max_frames:
        step = max(1, len(frame_files) // max_frames)
        frame_files = frame_files[::step][:max_frames]

    print(
        f"  Indexing {len(transcript_files)} transcript file(s) and "
        f"{len(frame_files)} frame image(s)."
    )
    return transcript_files + frame_files


def _lancedb_has_data(uri: str) -> bool:
    """
    Return True if both 'text_collection' and 'image_collection' LanceDB
    tables exist at *uri* and the text table contains at least one row.
    """
    try:
        import lancedb as _ldb
        db = _ldb.connect(uri)
        table_names = db.table_names()
        if "text_collection" in table_names and "image_collection" in table_names:
            text_table = db.open_table("text_collection")
            return text_table.count_rows() > 0
        return False
    except Exception as exc:
        print(f"  [lancedb check] Could not inspect tables: {exc}")
        return False


def build_vector_stores(lancedb_uri: str = None):
    """
    Create LanceDB vector stores for text and image collections.

    Args:
        lancedb_uri (str): LanceDB path/URI. Defaults to video_config.LANCEDB_URI.

    Returns:
        tuple: (text_store, image_store)
    """
    uri         = lancedb_uri or video_config.LANCEDB_URI
    text_store  = LanceDBVectorStore(uri=uri, table_name="text_collection")
    image_store = LanceDBVectorStore(uri=uri, table_name="image_collection")
    return text_store, image_store


def configure_embeddings(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Set the global LlamaIndex embedding model to a HuggingFace model.

    Args:
        model_name (str): HuggingFace model ID for embeddings.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_name, device="cuda")
    print(f"  Embedding model set: {model_name} (device: cuda)")


def build_video_index(
    data_folder:    str,
    lancedb_uri:    str  = None,
    force_reindex:  bool = False,
) -> MultiModalVectorStoreIndex:
    """
    Load documents from a folder (PNG frames + TXT transcript) and build
    a MultiModal LlamaIndex vector store index backed by LanceDB.

    Overflow Guard:
        If LanceDB already contains indexed data and ``force_reindex=False``
        (the default), the function reconnects to the existing index without
        re-embedding any documents.  This prevents duplicate vector entries
        on every pipeline run.

        Pass ``force_reindex=True`` (e.g. when the video changes or after a
        Force-Reset) to wipe the tables and rebuild from scratch.

    Args:
        data_folder   (str):  Folder with PNG frames + TXT transcript.
        lancedb_uri   (str):  Optional LanceDB URI override.
        force_reindex (bool): If True, always rebuild even if data exists.

    Returns:
        MultiModalVectorStoreIndex: Connected multimodal index.
    """
    configure_embeddings()
    uri = lancedb_uri or video_config.LANCEDB_URI

    # ── Guard: reconnect to existing index if already indexed ───────
    if not force_reindex and _lancedb_has_data(uri):
        print(
            "  LanceDB already contains indexed data — "
            "reconnecting to existing index (pass force_reindex=True to rebuild)."
        )
        text_store, image_store = build_vector_stores(uri)
        storage_context = StorageContext.from_defaults(
            vector_store=text_store,
            image_store=image_store,
        )
        # Reconnect without re-embedding: pass empty nodes list
        index = MultiModalVectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
        )
        print("  Existing MultiModal index loaded successfully.")
        return index

    # ── Fresh index (first run or force_reindex=True) ───────────────
    text_store, image_store = build_vector_stores(uri)
    storage_context = StorageContext.from_defaults(
        vector_store=text_store,
        image_store=image_store,
    )

    input_files = _select_index_files(data_folder, video_config.MAX_INDEXED_FRAMES)
    documents = SimpleDirectoryReader(input_files=input_files).load_data()
    print(f"  Loaded {len(documents)} documents from: {data_folder}")
    print("  Building multimodal embeddings and writing LanceDB index...")

    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    print("  MultiModal index built successfully.")
    return index


if __name__ == "__main__":
    import video_config as cfg
    index = build_video_index(cfg.OUTPUT_MIXED_DIR)
    retriever = index.as_retriever(similarity_top_k=1, image_similarity_top_k=3)
    print("  Retriever ready.")
