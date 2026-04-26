# -*- coding: utf-8 -*-
"""
text_processor.py
-----------------
Text loading, chunking, and embedding utilities for the Image-Text RAG pipeline.

Functions
---------
load_text_file(file_path) -> str
    Load raw text from a local .txt file using LangChain's TextLoader.

get_text_chunks(text) -> list[Document]
    Split raw text into overlapping chunks using CharacterTextSplitter.

build_embeddings() -> GoogleGenerativeAIEmbeddings
    Instantiate the Google Generative AI embedding model.

build_vectorstore(docs, embeddings) -> FAISS
    Create a FAISS vectorstore from a list of Document chunks.

build_retriever(vectorstore) -> VectorStoreRetriever
    Convert the FAISS vectorstore into a LangChain retriever.

Source: Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio_Yt (1).ipynb
  - get_text_chunks_langchain() (cell: fehnPFPGrnzJ)
  - TextLoader usage (cell: qLxSPRlMoa2E)
  - FAISS.from_documents() (cell: yi3NMD0pr_yI)
  - retriever.invoke() (cell: fEW4gvOlsJAQ)
"""

from langchain.schema.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from image_text_config import (
    EMBED_MODEL,
    TEXT_CHUNK_SIZE,
    TEXT_CHUNK_OVERLAP,
)


# ------------------------------------------------------------------
# Text Loading
# ------------------------------------------------------------------

def load_text_file(file_path: str) -> str:
    """
    Load and return the full text content of a .txt file.

    Parameters
    ----------
    file_path : Absolute or relative path to a plain-text file.

    Returns
    -------
    str — Full page content of the loaded document.
    """
    loader = TextLoader(file_path)
    docs = loader.load()
    return docs[0].page_content


# ------------------------------------------------------------------
# Text Chunking
# ------------------------------------------------------------------

def get_text_chunks(text: str) -> list:
    """
    Split ``text`` into overlapping Document chunks suitable for embedding.

    Uses CharacterTextSplitter with settings from image_text_config.py:
        chunk_size    = TEXT_CHUNK_SIZE
        chunk_overlap = TEXT_CHUNK_OVERLAP

    Parameters
    ----------
    text : Raw string to split.

    Returns
    -------
    list[Document]  — LangChain Document objects containing text chunks.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    print(f"[text_processor] Created {len(docs)} text chunks.")
    return docs


# ------------------------------------------------------------------
# Embeddings & Vector Store
# ------------------------------------------------------------------

def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Instantiate the Google Generative AI embedding model.

    Uses EMBED_MODEL defined in image_text_config.py.

    Returns
    -------
    GoogleGenerativeAIEmbeddings
    """
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def build_vectorstore(docs: list, embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    """
    Create and return a FAISS vectorstore from Document chunks.

    Parameters
    ----------
    docs       : List of LangChain Document objects.
    embeddings : GoogleGenerativeAIEmbeddings instance.

    Returns
    -------
    FAISS vectorstore.
    """
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    print(f"[text_processor] FAISS vectorstore built with {len(docs)} documents.")
    return vectorstore


def build_retriever(vectorstore: FAISS):
    """
    Convert a FAISS vectorstore into a LangChain retriever.

    Parameters
    ----------
    vectorstore : FAISS instance.

    Returns
    -------
    VectorStoreRetriever  — ready for use in RAG chains.
    """
    retriever = vectorstore.as_retriever()
    print("[text_processor] Retriever created from vectorstore.")
    return retriever


# ------------------------------------------------------------------
# Quick smoke-test (run: python text_processor.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import image_text_config  # noqa: F401 — ensures key is set before first use

    sample_text = (
        "Nike Dunk Low shoes are a classic basketball-inspired sneaker. "
        "Available in multiple colorways. Retail price approximately $110."
    )
    docs = get_text_chunks(sample_text)
    print(f"[smoke-test] Chunks: {[d.page_content for d in docs]}")

    embeddings = build_embeddings()
    vs = build_vectorstore(docs, embeddings)
    retriever = build_retriever(vs)
    results = retriever.invoke("Nike shoes price")
    print(f"[smoke-test] Retrieved: {results}")
