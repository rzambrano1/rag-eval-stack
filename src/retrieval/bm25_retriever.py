"""
A simple retriever class for the BM25 algorithm
"""

# Session Setup
# -------------

from typing import List, Dict, Any

# RAG modules
import pickle
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(
            self, 
            path: str
            ):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.chunks = payload["chunks"]
        self.bm25 = BM25Okapi(payload["tokenized_corpus"]) # Rebuild the index 

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"text": self.chunks[i], "score": float(scores[i]), "index": i}
            for i in top_indices
        ]
    
