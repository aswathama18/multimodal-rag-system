# -*- coding: utf-8 -*-
"""
pdf_rag_chain.py
----------------
Multimodal RAG chain for the PDF RAG pipeline.

Builds a LangChain pipeline that:
  1. Retrieves relevant text, table, and image content for a user question.
  2. Constructs a multimodal prompt (images + text).
  3. Generates a grounded answer using Gemini Pro.

Functions
---------
img_prompt_func(data_dict)      -> list[HumanMessage]
build_rag_chain(retriever)      -> Runnable
query(chain, question)          -> str
batch_query(chain, questions)   -> list[str]

Source: Document Multimodal Summrizer/rag_chain.py  (ported to PDF RAG folder)
"""

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

import pdf_config as config
from pdf_display import split_image_text_types


# ------------------------------------------------------------------
# Multimodal prompt builder
# ------------------------------------------------------------------

def img_prompt_func(data_dict: dict) -> list:
    """
    Build a multimodal HumanMessage from retrieved context and the user question.

    Parameters
    ----------
    data_dict : Dict with keys:
                  ``"context"``  – output of split_image_text_types()
                                   ({\"images\": [...], \"texts\": [...]})
                  ``"question"`` – the user's question string.

    Returns
    -------
    list[HumanMessage]  — Single-element list for the Gemini chat model.
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages: list = []

    # Add retrieved image(s)
    for image_b64 in data_dict["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })

    # Add text / question block
    messages.append({
        "type": "text",
        "text": (
            "You are a helpful research assistant.\n"
            "You will be given mixed information (text, tables, and/or images) "
            "retrieved from academic or technical documents.\n"
            "Use only this retrieved information to answer the user's question "
            "accurately and concisely. If the information is insufficient, say so.\n\n"
            f"User question: {data_dict['question']}\n\n"
            "Retrieved text / tables:\n"
            f"{formatted_texts}"
        ),
    })

    return [HumanMessage(content=messages)]


# ------------------------------------------------------------------
# RAG chain builder
# ------------------------------------------------------------------

def build_rag_chain(retriever):
    """
    Assemble the full multimodal RAG chain.

    Pipeline::
        question
          → retriever              (fetch relevant docs)
          → split_image_text_types (separate images from text)
          → img_prompt_func        (build multimodal prompt)
          → Gemini Pro             (generate grounded answer)
          → StrOutputParser        (return plain string)

    Parameters
    ----------
    retriever : LangChain retriever (typically MultiVectorRetriever from pdf_retriever.py).

    Returns
    -------
    Runnable  — invoke with a question string, returns an answer string.
    """
    model = ChatGoogleGenerativeAI(
        model=config.GEMINI_PRO_MODEL,
        temperature=0,
        max_tokens=2048,
    )

    chain = (
        {
            "context":  retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    print(f"[pdf_rag_chain] RAG chain built with model: {config.GEMINI_PRO_MODEL}")
    return chain


# ------------------------------------------------------------------
# Structured query helpers
# ------------------------------------------------------------------

def query(chain, question: str) -> str:
    """
    Run a single question through the multimodal RAG chain.

    Parameters
    ----------
    chain    : Runnable from build_rag_chain().
    question : Natural-language question string.

    Returns
    -------
    str  — Model answer grounded in retrieved context.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")

    print(f"\n[query] ❓ {question}")
    answer = chain.invoke(question.strip())
    print(f"[query] 💬 {answer}")
    return answer


def batch_query(chain, questions: list) -> list:
    """
    Run multiple questions through the chain sequentially.

    Parameters
    ----------
    chain     : Runnable from build_rag_chain().
    questions : List of question strings.

    Returns
    -------
    list[str]  — Answers in the same order as the input questions.
    """
    if not questions:
        return []

    answers = []
    total   = len(questions)
    for i, q in enumerate(questions, start=1):
        print(f"\n{'═'*60}")
        print(f"  Q {i}/{total}: {q}")
        print(f"{'═'*60}")
        answers.append(query(chain, q))
    return answers


# ------------------------------------------------------------------
# Standalone import check (run: python pdf_rag_chain.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import pdf_config  # noqa: F401
    print("pdf_rag_chain.py loaded successfully.")
    print(f"  RAG model: {config.GEMINI_PRO_MODEL}")
    print("  Use: chain = build_rag_chain(retriever)")
    print("       answer = query(chain, 'your question')")
