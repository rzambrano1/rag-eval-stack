"""
A simple retriever class for ChromaDB
"""

# Session Setup
# -------------

from typing import List, Dict, Any

# RAG modules
import chromadb
from sentence_transformers import SentenceTransformer

class ChromaRetriever:
    def __init__(
          self, 
          path: str, 
          collection_name: str, 
          embedding_model: str
          ):
        
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_collection(collection_name)
        self.encoder = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {
                "text": doc,
                "score": 1 - dist,  # convert distance to similarity
                #"metadata": meta   # no metadata in this example
            }
            for doc, dist in zip( # ommitted `meta`
                results["documents"][0],
                #results["metadatas"][0],
                results["distances"][0]
            )
        ]


# Retrieve function from scratch in tutorial available at:
# https://huggingface.co/blog/ngxson/make-your-own-rag
# ---------------------------------------------------------
# def retrieve(query, top_n=3):
#   query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
#   # temporary list to store (chunk, similarity) pairs
#   similarities = []
#   for chunk, embedding in VECTOR_DB:
#     similarity = cosine_similarity(query_embedding, embedding)
#     similarities.append((chunk, similarity))
#   # sort by similarity in descending order, because higher similarity means more relevant chunks
#   similarities.sort(key=lambda x: x[1], reverse=True)
#   # finally, return the top N most relevant chunks
#   return similarities[:top_n]