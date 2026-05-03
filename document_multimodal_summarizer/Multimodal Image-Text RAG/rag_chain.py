# -*- coding: utf-8 -*-
"""
rag_chain.py
------------
RAG chain builders for the Multimodal Image-Text RAG pipeline.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------
_RAG_TEMPLATE = """
```

{context}

```

{query}

Provide brief information and store location.
"""

_IMAGE_ONLY_TEMPLATE = """
Image information:
{image_info}

User question:
{query}

Answer the user's question using only the image information above.
"""

_HYBRID_TEMPLATE = """
Retrieved text context:
```
{context}
```

Image information:
{image_info}

User question:
{query}

Use the retrieved text context and image information together to answer the user's question.
"""


# ------------------------------------------------------------------
# Text RAG chain
# ------------------------------------------------------------------

def build_text_rag_chain(retriever, llm_text):
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
# Helpers
# ------------------------------------------------------------------

def get_text_query(x):
    return x["text_query"]


def _run_vision(llm_vision, inputs):
    from vision_query import query_vision_model

    mm_query = inputs.get("mm_query") or "Describe the image with details relevant to the user request."
    return query_vision_model(llm_vision, mm_query, inputs["image"])


# ------------------------------------------------------------------
# Image-only chain
# ------------------------------------------------------------------

def build_image_only_chain(llm_vision, llm_text):
    """
    Build an image-only route.

    The retriever is intentionally absent here: mm_query belongs to vision
    reasoning and must never be embedded or sent to FAISS.
    """
    prompt = ChatPromptTemplate.from_template(_IMAGE_ONLY_TEMPLATE)

    image_chain = (
        {
            "image_info": RunnableLambda(lambda x: _run_vision(llm_vision, x)),
            "query": RunnableLambda(lambda x: x.get("mm_query") or "What is shown in this image?"),
        }
        | prompt
        | llm_text
        | StrOutputParser()
    )

    print("[rag_chain] Image-only chain built.")
    return image_chain


# ------------------------------------------------------------------
# Full multimodal chain
# ------------------------------------------------------------------

def build_full_multimodal_chain(retriever, llm_vision, llm_text):
    """
    Build the hybrid text + image route.

    Data flow is deliberately split:
      - text_query is the only value sent to FAISS.
      - mm_query is the only query sent with the image to the vision model.
      - the final text LLM receives merged retrieved context + image_info.
    """
    prompt = ChatPromptTemplate.from_template(_HYBRID_TEMPLATE)

    full_chain = (
        {
            "context": RunnableLambda(get_text_query) | retriever,
            "image_info": RunnableLambda(lambda x: _run_vision(llm_vision, x)),
            "query": RunnableLambda(lambda x: f"{x['text_query']}\n\nImage-specific question: {x.get('mm_query', '')}")
        }
        | prompt
        | llm_text
        | StrOutputParser()
    )

    print("[rag_chain] Full multimodal chain built.")
    return full_chain
