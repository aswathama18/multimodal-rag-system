# _legacy

These files are the **original monolithic scripts** created during the initial
refactoring phase of the Multimodal RAG project.

They have been superseded by the modular sub-modules in:
  - PDF RAG/
  - Video RAG/
  - Multimodal Image-Text RAG/

## Contents

| File | Original Role |
|---|---|
| config.py | Shared config (Tesseract, Poppler, Gemini, Chroma) |
| main.py | Monolithic PDF pipeline script |
| extract_text.py | Text extraction from PDF elements |
| extract_tables.py | Table extraction from PDF elements |
| extract_images.py | Image extraction, encoding, summarisation |
| retriever.py | Chroma MultiVectorRetriever |
| rag_chain.py | Gemini Pro RAG chain |
| display.py | Jupyter/terminal image display utilities |
| pipeline.py | End-to-end pipeline entry point |
| verify_script.py | Standalone verification / test script |

> **Do not import from this folder in new code.**
> Use the modules inside PDF RAG/ instead.
