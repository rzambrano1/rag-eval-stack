"""
A simple evaluation pipeline to ingest corpora to be used in a RAG (Retrieval-Augmented Generation) system. 
"""

# Session Setup
# -------------

# Boilerplate Modules
import os
import sys
from pathlib import Path
import json

import time

import logging
from tqdm import tqdm

# Data wrangling and scientific computing modules
import numpy as np

# Config Files
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# Ingestion tooling
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Vector Databases

import chromadb
import lancedb
import pyarrow as pa
from rank_bm25 import BM25Okap

# Data Wrangling
# --------------

# Initializing Logger
logger = logging.getLogger(__name__)

# Loading the embedding model

# Embedding model from https://bge-model.com/Introduction/index.html a toolkit for search and RAG.
# Leaderboard available at https://huggingface.co/spaces/mteb/leaderboard 
model = SentenceTransformer("BAAI/bge-small-en-v1.5") # https://bge-model.com/bge/bge_v1_v1.5.html

def ingest_arxiv_files(langchain_text_spliter):

    logger.info('Preprocessing Arxiv files: chunking and embedding encoding... ')

    notebook_dir = Path.cwd()
    arxiv_dir = notebook_dir.parent / "raw" / "arxiv"

    arxiv_chunks = []

    for root, _, files in tqdm(os.walk(arxiv_dir), desc="Chunking and encoding entries from Arxiv...", colour='green'):
        
        for filename in tqdm(os.walk(files), desc="Looping entries in Arxiv topic...", colour='blue', leave=False):  # loop through files in the current directory

            file_path = os.path.join(root, filename)
            file_topic = filename.split(".")[0].replace("_", " ")
            
            with open(file_path, 'r') as file:
                
                data = json.load(file)

                for entry in data:

                    print(entry.keys())
                    # parsed_data = json.dumps(entry, indent=4)
                    entry_text = entry['summary']

                    print(file_topic)
                    print(file_path)
                    print(entry_text)

                    topic_entry_chunks = langchain_text_spliter.split_text(entry_text)

                    print("-"*50)

                    for i, chunk in enumerate(topic_entry_chunks):
                        arxiv_chunks.append(chunk)
                        print(f"Chunk {i+1}: {chunk}\n")

    arxiv_chunk_embeddings = model.encode(
        arxiv_chunks,
        normalize_embeddings=True,  # important for cosine similarity
        show_progress_bar=True
    )

    assert len(arxiv_chunks) == len(arxiv_chunk_embeddings), "Oh No! The length of the Arxiv chunks are not matching embeddings..."

    return arxiv_chunks, arxiv_chunk_embeddings

def ingest_wiki_files(langchain_text_spliter):

    logger.info('Preprocessing Wikipedia files: chunking and embedding encoding... ')

    notebook_dir = Path.cwd()
    wiki_dir = notebook_dir.parent / "raw" / "wikipedia"

    wiki_chunks = []

    for root, _, files in tqdm(os.walk(wiki_dir), desc="Chunking and encoding entries from Wikipedia...", colour='green'):
        for filename in files:  # loop through files in the current directory

            file_path = os.path.join(root, filename)
            file_topic = filename.split(".")[0].replace("_", " ")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                
                entry_text = json.load(file)

                print('topic: ', file_topic)
                print('path: ',file_path)
                print('entry: ', entry_text)

                topic_entry_chunks = langchain_text_spliter.split_text(entry_text)

                print("-"*50)

                for i, chunk in enumerate(topic_entry_chunks):
                    wiki_chunks.append(chunk)
                    print(f"Chunk {i+1}: {chunk}\n")

    wiki_chunk_embeddings = model.encode(
        wiki_chunks,
        normalize_embeddings=True,  # important for cosine similarity
        show_progress_bar=True
    )

    assert len(wiki_chunks) == len(wiki_chunk_embeddings), "Oh No! The length of the Wikipedia chunks are not matching embeddings..."

    return wiki_chunks, wiki_chunk_embeddings

def ingest_firecrawl_files(langchain_text_spliter):

    logger.info('Preprocessing Firecrawl files: chunking and embedding encoding... ')

    notebook_dir = Path.cwd()

    firecrawl_dir = notebook_dir.parent / "raw" / "firecrawl"

    firecrawl_chunks = []

    for root, _, files in tqdm(os.walk(firecrawl_dir), desc="Chunking and encoding entries from Firecrawl...", colour='green'):

        for filename in files:  # loop through files in the current directory

            file_path = os.path.join(root, filename)
            file_topic = filename.split(".")[0].replace("_", " ")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                
                entry = json.load(file)

                print(entry)
                #parsed_data = json.dumps(entry, indent=4)
                entry_data = entry['data']

                print('topic: ', file_topic)
                print('path: ',file_path)
                print('entry: ', entry_data)

                for data in entry_data:

                    entry_text = data['markdown']
                    print(entry_text)
                    
                    topic_entry_chunks = langchain_text_spliter.split_text(entry_text)

                    print("-"*50)

                    for i, chunk in enumerate(topic_entry_chunks):
                        firecrawl_chunks.append(chunk)
                        print(f"Chunk {i+1}: {chunk}\n")

    firecrawl_chunk_embeddings = model.encode(
        firecrawl_chunks,
        normalize_embeddings=True,  # important for cosine similarity
        show_progress_bar=True
    )

    assert len(firecrawl_chunks) == len(firecrawl_chunk_embeddings), "Oh No! The length of the Firecrawl chunks are not matching embeddings..."

    return firecrawl_chunks, firecrawl_chunk_embeddings

def ingest_firecrawl_files(langchain_text_spliter, bank):
    
    if bank not in ["JPM", "COF"]:
        raise ValueError("Allowed values for bank are: ['JPM', 'COF']")

    logger.info(f'Preprocessing SEC 10-K filings files for {bank}: chunking and embedding encoding... ')

    notebook_dir = Path.cwd()

    sec_dir = notebook_dir.parent / "raw" / "sec" 
    txt_files = list(sec_dir.rglob("*.txt"))

    bank_files = [file for file in txt_files if bank in str(file)]

    for file_path in bank_files:

        with open(file_path, 'r', encoding='utf-8') as file:
            
            text_content = file.read()
            print(f'10-K file submission length: {text_content}')
            print(text_content[100])

            bank_entry_chunks = langchain_text_spliter.split_text(text_content)

            bank_chunk_embeddings = model.encode(
                bank_entry_chunks,
                normalize_embeddings=True,  # important for cosine similarity
                show_progress_bar=True
            )
    
    assert len(bank_entry_chunks) == len(bank_chunk_embeddings), "Oh No! The length of the SEC 10-K filings chunks are not matching embeddings..."

    return bank_entry_chunks, bank_chunk_embeddings

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

    # Chunking and embeddings

    arxiv_chunks, arxiv_chunk_embeddings = ingest_arxiv_files(text_splitter)
    wiki_chunks, wiki_chunk_embeddings = ingest_wiki_files(text_splitter)
    firecrawl_chunks, firecrawl_chunk_embeddings = ingest_firecrawl_files(text_splitter)

    jpm_chunks, jmp_embeddings = ingest_firecrawl_files(text_splitter, "JPM")
    cof_chunks, cof_embeddings = ingest_firecrawl_files(text_splitter, "COF")

    combined_chunks = arxiv_chunks + wiki_chunks + firecrawl_chunks + jpm_chunks + cof_chunks
    combined_embeddings = arxiv_chunk_embeddings + wiki_chunk_embeddings + firecrawl_chunk_embeddings + jmp_embeddings + cof_embeddings
    
    # Vector Databases

    root_dir = Path.cwd()
    chroma_dir = os.path.normpath(root_dir / cfg.ingest.CHROMA_DIR)
    


if __name__ == "__main__":
    main()