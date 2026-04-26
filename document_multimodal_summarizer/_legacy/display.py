# -*- coding: utf-8 -*-
"""
display.py
----------
Utilities to render and inspect retrieved multimodal content.

Automatically detects whether the code is running inside a
Jupyter / Colab kernel or a plain terminal:

- **Jupyter / Colab** : images are rendered inline as HTML.
- **Terminal**        : images are saved to ``config.DEBUG_IMAGE_DIR``
                        and the file path is printed, so you can open
                        them manually to verify pipeline output.

Public API:
    plt_img_base64(img_base64, label)       -> None
    looks_like_base64(sb)                   -> bool
    is_image_data(b64data)                  -> bool
    resize_base64_image(base64_string, size)-> str
    split_image_text_types(docs)            -> dict
    display_retrieval_results(docs)         -> None
"""

# ------------------------------------------------------------------
# Standard library
# ------------------------------------------------------------------
import base64
import io
import os
import re
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Third-party — image processing
# ------------------------------------------------------------------
from PIL import Image

# ------------------------------------------------------------------
# LangChain — document type detection
# ------------------------------------------------------------------
from langchain_core.documents import Document

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
import config

# ------------------------------------------------------------------
# Optional IPython display (only available inside Jupyter/Colab)
# ------------------------------------------------------------------
try:
    from IPython.display import HTML, display as _ipython_display
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False


# ── Jupyter / terminal detection ───────────────────────────────────

def _is_jupyter() -> bool:
    """Return True when running inside a Jupyter or Colab kernel."""
    try:
        shell_name = get_ipython().__class__.__name__  # type: ignore[name-defined]  # noqa: F821
        return shell_name in ("ZMQInteractiveShell", "google.colab._shell")
    except NameError:
        return False


# ── Debug image directory (terminal fallback) ──────────────────────

_DEBUG_DIR = Path(config.DEBUG_IMAGE_DIR)


def _ensure_debug_dir() -> Path:
    """Create config.DEBUG_IMAGE_DIR if it does not already exist."""
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    return _DEBUG_DIR


# ══════════════════════════════════════════════════════════════════
#  Core display helper
# ══════════════════════════════════════════════════════════════════

def plt_img_base64(img_base64: str, label: str = "image") -> None:
    """
    Display a base64-encoded image.

    - **Jupyter / Colab**: renders the image inline as HTML.
    - **Terminal**       : decodes and saves the image to
                           ``config.DEBUG_IMAGE_DIR/<label>_<timestamp>.jpg``
                           then prints the absolute path.

    Args:
        img_base64 : Base64-encoded JPEG or PNG string.
        label      : Optional label used in the saved filename (terminal mode).
    """
    if _is_jupyter() and _IPYTHON_AVAILABLE:
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        _ipython_display(HTML(image_html))
    else:
        # ── Terminal fallback: save image to disk ────────────────
        debug_dir = _ensure_debug_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename  = f"{label}_{timestamp}.jpg"
        out_path  = debug_dir / filename

        try:
            img_data = base64.b64decode(img_base64)
            img      = Image.open(io.BytesIO(img_data))
            img.save(str(out_path), format="JPEG")
            print(f"  [display] Image saved → {out_path.resolve()}")
        except Exception as exc:
            print(f"  [display] WARNING: Could not save image '{label}': {exc}")


# ══════════════════════════════════════════════════════════════════
#  Base64 / image-type detection helpers
# ══════════════════════════════════════════════════════════════════

def looks_like_base64(sb: str) -> bool:
    """
    Return True if *sb* looks like a base64-encoded string.

    Checks only the character set; does not decode.

    Args:
        sb : String to test.

    Returns:
        bool
    """
    if not isinstance(sb, str) or len(sb) < 16:
        return False
    return bool(re.match(r"^[A-Za-z0-9+/]+[=]{0,2}$", sb))


def is_image_data(b64data: str) -> bool:
    """
    Return True if the base64 payload decodes to a recognised image format.

    Detects JPEG, PNG, GIF, and WebP by inspecting magic bytes.

    Args:
        b64data : Base64 string to test.

    Returns:
        bool
    """
    _SIGNATURES = {
        b"\xFF\xD8\xFF":                      "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38":                  "gif",
        b"\x52\x49\x46\x46":                  "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in _SIGNATURES)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
#  Image resizing utility
# ══════════════════════════════════════════════════════════════════

def resize_base64_image(base64_string: str, size: tuple = (1300, 600)) -> str:
    """
    Resize a base64-encoded image and return the result as base64.

    Args:
        base64_string : Original base64 image string.
        size          : Target ``(width, height)`` in pixels.

    Returns:
        str: Base64-encoded resized image (same format: JPEG/PNG/…).
    """
    img_data    = base64.b64decode(base64_string)
    img         = Image.open(io.BytesIO(img_data))
    orig_format = img.format or "JPEG"

    resized = img.resize(size, Image.LANCZOS)

    buffer = io.BytesIO()
    resized.save(buffer, format=orig_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════
#  Split retrieved documents into images vs. texts
# ══════════════════════════════════════════════════════════════════

def split_image_text_types(docs: list) -> dict:
    """
    Partition retrieved documents into base64 images and plain texts.

    Base64 image strings are resized to ``(1300 × 600)`` pixels for
    uniform display; all other content is treated as text / table HTML.

    Args:
        docs : List of raw strings or LangChain ``Document`` objects,
               typically returned by ``retriever.invoke(query)``.

    Returns:
        dict with keys:
            ``"images"`` (``list[str]``) — resized base64 image strings
            ``"texts"``  (``list[str]``) — plain text / table HTML strings
    """
    b64_images: list = []
    texts:      list = []

    for doc in docs:
        # LangChain Document objects: extract the page_content string
        content = doc.page_content if isinstance(doc, Document) else str(doc)

        if looks_like_base64(content) and is_image_data(content):
            b64_images.append(resize_base64_image(content, size=(1300, 600)))
        else:
            texts.append(content)

    return {"images": b64_images, "texts": texts}


# ══════════════════════════════════════════════════════════════════
#  High-level: display all retrieved docs at once
# ══════════════════════════════════════════════════════════════════

def display_retrieval_results(docs: list) -> None:
    """
    Display every document returned by the retriever in a readable format.

    - Images → rendered inline (Jupyter) or saved to disk (terminal).
    - Text / table content → printed to stdout with a 500-char preview.

    Args:
        docs : Output from ``retriever.invoke(query)``
               (list of strings or ``Document`` objects).
    """
    if not docs:
        print("  [display] No documents retrieved.")
        return

    split = split_image_text_types(docs)

    # ── Images ────────────────────────────────────────────────────
    if split["images"]:
        print(f"\n{'─'*60}")
        print(f"  Retrieved {len(split['images'])} image(s):")
        print(f"{'─'*60}")
        for i, img_b64 in enumerate(split["images"]):
            print(f"  [Image {i + 1}]")
            plt_img_base64(img_b64, label=f"result_img_{i + 1}")
    else:
        print("  [No images in retrieved results]")

    # ── Texts / Tables ─────────────────────────────────────────────
    if split["texts"]:
        print(f"\n{'─'*60}")
        print(f"  Retrieved {len(split['texts'])} text/table chunk(s):")
        print(f"{'─'*60}")
        for i, txt in enumerate(split["texts"]):
            preview = txt[:500] + ("…" if len(txt) > 500 else "")
            print(f"\n  [Chunk {i + 1}]\n  {preview}")
    else:
        print("  [No text chunks in retrieved results]")


# ══════════════════════════════════════════════════════════════════
#  Standalone smoke-test   (python display.py)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("display.py smoke-test")
    print(f"  Debug image dir : {config.DEBUG_IMAGE_DIR}")
    print(f"  Jupyter mode    : {_is_jupyter()}")

    # Test text rendering
    sample_docs = [
        "This is a sample retrieved text chunk about Retrieval-Augmented Generation.",
        "RAG combines a dense retriever with a seq2seq generator fine-tuned end-to-end.",
    ]
    print("\n--- display_retrieval_results (text-only) ---")
    display_retrieval_results(sample_docs)
    print("\nSmoke-test complete.")
