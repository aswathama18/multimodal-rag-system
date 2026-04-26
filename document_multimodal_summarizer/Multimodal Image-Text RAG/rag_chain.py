# -*- coding: utf-8 -*-
"""
rag_chain.py
------------
RAG chain builders for the Multimodal Image-Text RAG pipeline.

Functions
---------
build_text_rag_chain(retriever, llm_text) -> Runnable
    Build a text-context RAG chain: retriever → prompt → LLM → string output.

build_full_multimodal_chain(rag_chain, llm_vision) -> Runnable
    Build a vision-first full chain: vision model describes an image,
    then that description is passed through the text RAG chain.

Source: Multimodal_RAG_with_Gemini_Langchain_and_Google_AI_Studio_Yt (1).ipynb
  - rag_chain definition (cell: LummLlRtsf3p)
  - full_chain definition (cell: US0kn6zFs63Z)
  - prompt template      (cell: FovmYztwsVPh)
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# ------------------------------------------------------------------
# Prompt template  (matches the notebook)
# ------------------------------------------------------------------
_RAG_TEMPLATE = """
```
{context}
```

{query}


Provide brief information and store location.
"""


# ------------------------------------------------------------------
# Text RAG chain
# ------------------------------------------------------------------

def build_text_rag_chain(retriever, llm_text):
    """
    Build a standard text RAG chain.

    Pipeline
    --------
    1. ``retriever``     — fetches relevant Document chunks for the user query.
    2. ``ChatPromptTemplate`` — injects retrieved context + query into the prompt.
    3. ``llm_text``      — Gemini text model generates the answer.
    4. ``StrOutputParser`` — strips the AIMessage wrapper, returning a plain string.

    Parameters
    ----------
    retriever : LangChain retriever (from text_processor.build_retriever).
    llm_text  : ChatGoogleGenerativeAI text model (from model_loader.load_text_model).

    Returns
    -------
    Runnable  — invoke with a query string, returns an answer string.
    """
    prompt = ChatPromptTemplate.from_template(_RAG_TEMPLATE)

    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm_text
        | StrOutputParser()
    )
    print("[rag_chain] Text RAG chain built.")
    return rag_chain


# ------------------------------------------------------------------
# Full multimodal chain (vision → text RAG)
# ------------------------------------------------------------------

def build_full_multimodal_chain(rag_chain, llm_vision):
    """
    Build the end-to-end multimodal chain.

    Pipeline
    --------
    1. ``llm_vision``    — Gemini vision model processes an image+text HumanMessage
                           and returns a textual description.
    2. ``StrOutputParser`` — converts AIMessage to plain text.
    3. ``rag_chain``     — text RAG chain retrieves product info and generates final answer.

    Parameters
    ----------
    rag_chain   : Runnable from build_text_rag_chain().
    llm_vision  : ChatGoogleGenerativeAI vision model (from model_loader.load_vision_model).

    Returns
    -------
    Runnable  — invoke with a list containing a HumanMessage (image + prompt),
                returns a final answer string.

    Example
    -------
    >>> from langchain_core.messages import HumanMessage
    >>> message = HumanMessage(content=[
    ...     {"type": "text", "text": "Describe this product."},
    ...     {"type": "image_url", "image_url": pil_image},
    ... ])
    >>> answer = full_chain.invoke([message])
    """
    full_chain = (
        RunnablePassthrough()
        | llm_vision
        | StrOutputParser()
        | rag_chain
    )
    print("[rag_chain] Full multimodal chain built.")
    return full_chain
