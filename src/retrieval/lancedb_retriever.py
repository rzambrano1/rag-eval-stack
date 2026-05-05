"""
A simple retriever class for LanceDB
"""

# Session Setup
# -------------

from typing import List, Dict, Any

# RAG modules
import lancedb
from sentence_transformers import SentenceTransformer

class LanceDBRetriever:
    def __init__(
            self, 
            path: str, 
            table_name: str, 
            embedding_model: str
            ):
        self.db = lancedb.connect(path)
        self.table = self.db.open_table(table_name)
        self.encoder = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.encoder.encode(query).tolist()
        results = (
            self.table.search(query_vec)
            .limit(top_k)
            .to_pandas()
        )
        return [
            {
                "text": row["documents"],
                "score": 1 - row["_distance"],
                # "metadata": row.get("source", "") # no metadata in this example
            }
            for _, row in results.iterrows()
        ]
    


