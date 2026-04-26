# -*- coding: utf-8 -*-
"""
retriever.py
------------
MultiVector Retriever for the Multimodal RAG pipeline.

Indexes summaries of texts, tables, and images into a **disk-persistent**
Chroma vector store, but maps each summary back to the original raw content
so the retriever returns source material (text / HTML table / base64 image).

Public API:
    build_vectorstore(collection_name, persist_directory) -> Chroma
    create_multi_vector_retriever(vectorstore, ...) -> MultiVectorRetriever

Usage:
    from retriever import build_vectorstore, create_multi_vector_retriever

    vectorstore = build_vectorstore()
    retriever   = create_multi_vector_retriever(
        vectorstore,
        text_summaries,  texts,
        table_summaries, tables,
        image_summaries, img_base64_list,
    )
    docs = retriever.invoke("your query here")
"""

# ------------------------------------------------------------------
# Standard library
# ------------------------------------------------------------------
import uuid

# ------------------------------------------------------------------
# LangChain — vector store, retriever, docstore
# NOTE: In LangChain >= 1.x, MultiVectorRetriever and InMemoryStore
# live in langchain_classic (not langchain or langchain_community).
# ------------------------------------------------------------------
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ------------------------------------------------------------------
# Google Generative AI — embeddings
# ------------------------------------------------------------------
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ------------------------------------------------------------------
# Config — sets GOOGLE_API_KEY, model names, paths (side-effects)
# ------------------------------------------------------------------
import config  # noqa: F401


# ══════════════════════════════════════════════════════════════════
#  Build Chroma vector store with disk persistence
# ══════════════════════════════════════════════════════════════════

def build_vectorstore(
    collection_name:   str = config.CHROMA_COLLECTION_NAME,
    persist_directory: str = config.CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    Initialise (or reload from disk) a Chroma vector store backed by
    Google Gemini text embeddings.

    The store persists to ``persist_directory`` so the index survives
    process restarts — critical for deployment and avoiding re-embedding
    costs on every run.

    Args:
        collection_name   : Chroma collection identifier (default from config).
        persist_directory : On-disk directory for the index (default from config).

    Returns:
        Chroma: Vector store instance, loaded from disk if it already exists.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


# ══════════════════════════════════════════════════════════════════
#  Internal helper: add (summary, raw-content) pairs to the store
# ══════════════════════════════════════════════════════════════════

def _add_documents(
    retriever:     MultiVectorRetriever,
    doc_summaries: list,
    doc_contents:  list,
    id_key:        str,
) -> None:
    """
    Embed *summaries* into the vector store and link each to its
    raw *content* in the docstore via a shared UUID.

    Args:
        retriever     : The MultiVectorRetriever to populate.
        doc_summaries : Summaries to embed (what semantic search sees).
        doc_contents  : Raw originals linked 1-to-1 with summaries
                        (what the retriever returns on a hit).
        id_key        : Metadata key that connects summary ↔ raw content.

    Raises:
        ValueError: If lengths of summaries and contents differ.
    """
    if len(doc_summaries) != len(doc_contents):
        raise ValueError(
            f"Length mismatch: {len(doc_summaries)} summaries vs "
            f"{len(doc_contents)} contents. "
            "Each summary must map to exactly one piece of raw content."
        )

    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(doc_summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


# ══════════════════════════════════════════════════════════════════
#  Main public function
# ══════════════════════════════════════════════════════════════════

def create_multi_vector_retriever(
    vectorstore:     Chroma,
    text_summaries:  list,
    texts:           list,
    table_summaries: list,
    tables:          list,
    image_summaries: list,
    images:          list,
) -> MultiVectorRetriever:
    """
    Build a MultiVectorRetriever that searches over *summaries* but
    returns the corresponding *raw* content (text / HTML table / base64 image).

    Args:
        vectorstore     : Pre-built Chroma instance (from build_vectorstore()).
        text_summaries  : LLM-generated summaries of each text chunk.
        texts           : Raw text chunks (same order as text_summaries).
        table_summaries : LLM-generated summaries of each table.
        tables          : Raw table HTML strings (same order as table_summaries).
        image_summaries : LLM-generated summaries of each image.
        images          : Base64-encoded image strings (same order as image_summaries).

    Returns:
        MultiVectorRetriever: Ready-to-query retriever.

    Notes:
        - Empty summary lists are safely skipped (no error).
        - The in-memory docstore is rebuilt every run; only the *vector index*
          persists to disk via Chroma.  For full docstore persistence across
          restarts, replace InMemoryStore with a Redis or SQLite-backed store.
    """
    store  = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    if text_summaries:
        _add_documents(retriever, text_summaries, texts, id_key)

    if table_summaries:
        _add_documents(retriever, table_summaries, tables, id_key)

    if image_summaries:
        _add_documents(retriever, image_summaries, images, id_key)

    return retriever


# ══════════════════════════════════════════════════════════════════
#  Standalone smoke-test   (python retriever.py)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Building vector store …")
    vs = build_vectorstore()

    dummy_texts   = ["The sky is blue."]
    dummy_t_summ  = ["A statement about sky colour."]
    dummy_tabs    = ["<table><tr><td>A</td></tr></table>"]
    dummy_tab_sum = ["A simple one-cell HTML table."]

    print("Creating retriever …")
    ret = create_multi_vector_retriever(
        vs,
        dummy_t_summ, dummy_texts,
        dummy_tab_sum, dummy_tabs,
        [], [],  # no images in smoke-test
    )

    results = ret.invoke("sky colour")
    print(f"Retrieval returned {len(results)} document(s).")
    for doc in results:
        print(" •", doc)