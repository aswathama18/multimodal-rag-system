# -*- coding: utf-8 -*-
"""
pdf_retriever.py
----------------
MultiVector Retriever for the PDF Multimodal RAG pipeline.

Indexes summaries of texts, tables, and images into a disk-persistent
Chroma vector store, mapping each summary back to its raw content so
the retriever returns source material (text / HTML table / base64 image).

Functions
---------
build_vectorstore(collection_name, persist_directory) -> Chroma
    Initialise or reload a Chroma vectorstore backed by Gemini embeddings.

create_multi_vector_retriever(vectorstore, ...) -> MultiVectorRetriever
    Build a retriever that searches summaries but returns raw originals.

Source: Document Multimodal Summrizer/retriever.py  (ported to PDF RAG folder)
"""

import uuid

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import pdf_config as config


# ------------------------------------------------------------------
# Build Chroma vector store
# ------------------------------------------------------------------

def build_vectorstore(
    collection_name:   str = config.CHROMA_COLLECTION_NAME,
    persist_directory: str = config.CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    Initialise (or reload from disk) a Chroma vector store.

    Parameters
    ----------
    collection_name   : Chroma collection identifier (default from pdf_config).
    persist_directory : On-disk directory for the Chroma index.

    Returns
    -------
    Chroma  — Vector store instance, reloaded from disk if it already exists.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    print(f"[pdf_retriever] Chroma vectorstore ready at: {persist_directory}")
    return vectorstore


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _add_documents(
    retriever:     MultiVectorRetriever,
    doc_summaries: list,
    doc_contents:  list,
    id_key:        str,
) -> None:
    """
    Embed summaries into the vector store and link each to its raw content
    in the docstore via a shared UUID.

    Raises ValueError if len(doc_summaries) != len(doc_contents).
    """
    if len(doc_summaries) != len(doc_contents):
        raise ValueError(
            f"Length mismatch: {len(doc_summaries)} summaries vs "
            f"{len(doc_contents)} contents."
        )

    doc_ids      = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(doc_summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


# ------------------------------------------------------------------
# Main public function
# ------------------------------------------------------------------

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
    Build a MultiVectorRetriever that searches over summaries but returns
    the corresponding raw content.

    Parameters
    ----------
    vectorstore     : Chroma instance from build_vectorstore().
    text_summaries  : LLM-generated summaries of each text chunk.
    texts           : Raw text chunks (same order as text_summaries).
    table_summaries : LLM-generated summaries of each table.
    tables          : Raw table HTML strings (same order as table_summaries).
    image_summaries : LLM-generated summaries of each image.
    images          : Base64-encoded image strings (same order as image_summaries).

    Returns
    -------
    MultiVectorRetriever  — Ready-to-query retriever.

    Notes
    -----
    Empty summary lists are safely skipped.
    The in-memory docstore is rebuilt every run; only the vector index
    persists via Chroma.
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

    print("[pdf_retriever] MultiVectorRetriever created.")
    return retriever


# ------------------------------------------------------------------
# Quick smoke-test (run: python pdf_retriever.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import pdf_config  # noqa: F401

    print("Building Chroma vectorstore …")
    vs = build_vectorstore()

    print("Creating multi-vector retriever with dummy data …")
    ret = create_multi_vector_retriever(
        vs,
        ["A statement about sky colour."], ["The sky is blue."],
        ["A simple one-cell HTML table."], ["<table><tr><td>A</td></tr></table>"],
        [], [],
    )

    results = ret.invoke("sky colour")
    print(f"[smoke-test] Retrieved {len(results)} document(s).")
    for doc in results:
        print(" •", doc)
