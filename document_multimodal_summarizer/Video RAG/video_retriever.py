# -*- coding: utf-8 -*-
"""
video_retriever.py
------------------
Retrieve relevant frames and text from the MultiModal index, and
provide visualization utilities.

Usage:
    from video_retriever import retrieve_multimodal, plot_images
"""

import os
import matplotlib.pyplot as plt
from PIL import Image

from llama_index.core.schema import ImageNode
# display_source_node removed: requires IPython (notebook-only) and is unused in server/CLI context


def retrieve_multimodal(retriever_engine, query_str):
    """
    Retrieve top-k images and text relevant to the query from the engine.

    Args:
        retriever_engine: LlamaIndex MultiModal retriever engine.
        query_str (str):  The search query.

    Returns:
        tuple: (retrieved_images, retrieved_texts)
            - retrieved_images (list[str]): File paths to the retrieved images.
            - retrieved_texts  (list[str]): The matched text snippets.
    """
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_images = []
    retrieved_texts  = []

    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_images.append(res_node.node.metadata["file_path"])
        else:
            # Uncomment below if running in a notebook context
            # display_source_node(res_node, source_length=200)
            retrieved_texts.append(res_node.text)

    return retrieved_images, retrieved_texts


def plot_images(images_path, max_images=5):
    """
    Visualize retrieved images using matplotlib.

    Args:
        images_path (list[str]): File paths of images to plot.
        max_images (int): Maximum number of images to display.
    """
    images_shown = 0
    plt.figure(figsize=(16, 9))

    for img_path in images_path:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= max_images:
                break
    
    plt.tight_layout()
    plt.show()
