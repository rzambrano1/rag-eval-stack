"""
A simple evaluation pipeline to ingest corpora to be used in a RAG (Retrieval-Augmented Generation) system. 
"""

# Session Setup

# Boilerplate Modules
import os
import sys
import json

import time

from tqdm import tqdm

# Config Files
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# Ingestion tooling
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Loading the embedding model

# Embedding model from https://bge-model.com/Introduction/index.html a toolkit for search and RAG.
# Leaderboard available at https://huggingface.co/spaces/mteb/leaderboard 
model = SentenceTransformer("BAAI/bge-small-en-v1.5") # https://bge-model.com/bge/bge_v1_v1.5.html

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg : DictConfig):

    # Loading LangChain for chunking
    # -- chunk_size: max characters per chunk
    # -- chunk_overlap: overlapping characters to keep context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.ingest.chunk_size,
        chunk_overlap=cfg.ingest.chunk_overlap,
        length_function=cfg.ingest.length_function,
    )

if __name__ == "__main__":
    main()