"""
A simple evaluation pipeline to evaluate a RAG (Retrieval-Augmented Generation) system. 
"""

# Session Setup

import wikipedia

pages = ["Attention Is All You Need", "BERT (language model)", 
         "GPT-4", "Retrieval-augmented generation", "LoRA (machine learning)"]
docs = [wikipedia.page(p).content for p in pages]