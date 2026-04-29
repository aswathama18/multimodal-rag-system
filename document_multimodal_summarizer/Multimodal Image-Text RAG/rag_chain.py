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
# Prompt template
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
# Helper: Extract text query
# ------------------------------------------------------------------

def get_text_query(x):
    return x["text_query"]


# ------------------------------------------------------------------
# Full multimodal chain
# ------------------------------------------------------------------

def build_full_multimodal_chain(rag_chain, retriever, llm_vision):

    # ----------------------------
    # Vision processing (image → text)
    # ----------------------------
    def run_vision(x):
        from langchain_core.messages import HumanMessage
        import base64
        from io import BytesIO

        # Convert PIL → base64
        buf = BytesIO()
        x["image"].save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        message = HumanMessage(
            content=[
                {"type": "text", "text": x["text_query"]},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{img_b64}",
                },
            ]
        )

        return llm_vision.invoke([message]).content


    # ----------------------------
    # Combine query + image context
    # ----------------------------
    def build_prompt(x):
        return {
            "query": f"{x['text_query']}\n\nImage context: {x['image_info']}",
            "context": x["context"],
        }


    # ----------------------------
    # Full chain
    # ----------------------------
    full_chain = (
        {
            "context": RunnableLambda(get_text_query) | retriever,
            "image_info": RunnableLambda(run_vision),
            "text_query": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | rag_chain
    )

    print("[rag_chain] Full multimodal chain built.")
    return full_chain
