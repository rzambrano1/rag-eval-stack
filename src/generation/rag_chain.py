"""
A simple RAG chain script
"""

# Session Setup
# -------------

# Boilerplate Modules
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from typing import List, Dict, Any, Literal

curr_dir = Path(__file__).resolve().parent 
root_dir = curr_dir.parent.parent
# To be able to import the retrievers
src_dir = curr_dir.parent                   
sys.path.insert(0, str(src_dir / "retrieval"))

# Import [golden] retriever classes [woof!]
from bm25_retriever import BM25Retriever
from chroma_retriever import ChromaRetriever
from lancedb_retriever import LanceDBRetriever

# LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Load variables from .env into the environment
load_dotenv()

CONFIG = {
    "db_directory" : {
        "chroma" : root_dir / "vector_stores" / "chroma",
        "lancedb"  : root_dir / "vector_stores" / "lancedb",
        "bm25"   : root_dir / "vector_stores" / "bm25",
    },
    "collection_name" : "rag_corpus",
    "embedding_model": "BAAI/bge-small-en-v1.5",
}

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY 
the context provided below. If the answer is not in the context, say 
"I don't have enough information to answer this question."

Context:
{context}
"""

class RAGChain:
    def __init__(
        self,
        retriever_type: Literal["bm25", "chroma", "lancedb"],
        retriever_kwargs: dict,
        model: str = "gemini-2.5-flash",
        top_k: int = 5,
    ):
        self.model = model
        self.top_k = top_k

        # Initializing LLM model for prototyping
        self.client =  ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_KEY")
            )
        
        self.retriever = self._build_retriever(retriever_type, retriever_kwargs)

    def _build_retriever(self, retriever_type: str, kwargs: dict):
        if retriever_type == "bm25":
            return BM25Retriever(**kwargs)
        elif retriever_type == "chroma":
            return ChromaRetriever(**kwargs)
        elif retriever_type == "lancedb":
            return LanceDBRetriever(**kwargs)
        else:
            raise ValueError(f'Oh No! >>> unknown retriever type: {retriever_type}...\n...Valid values are: ["bm25", "chroma", "lancedb"]')

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        return "\n\n---\n\n".join(
            f"[Chunk {i+1}]\n{c['text']}" for i, c in enumerate(chunks)
        )

    def run(self, query: str) -> Dict[str, Any]:
        chunks = self.retriever.retrieve(query, top_k=self.top_k)
        context = self._format_context(chunks)

        response = self.client.invoke(
            SYSTEM_PROMPT.format(context=context) + f"\n\nQuestion: {query}"
        )

        return {
            "query": query,
            "answer": response.content,
            "retrieved_chunks": chunks,
            "context": context,
        }


if __name__ == "__main__":

    flag = True
    
    while flag:
        db_name = input('\nEnter the DB to test ["bm25", "chroma", "lancedb"]: ')
        if db_name in ["bm25", "chroma", "lancedb"]:
            flag = False
        else:
            print('Oh No! Invalid input: Select a valid database name, the input is case sensitive...')
            continue
    
    if db_name == "bm25":
        retriever_kwargs = {
            "path": CONFIG["db_directory"]["bm25"] / "bm25_payload.pkl"
        }
    elif db_name == "chroma":
        retriever_kwargs = {
            "path": str(CONFIG["db_directory"]["chroma"]),
            "collection_name": CONFIG["collection_name"],
            "embedding_model": CONFIG["embedding_model"],
        }
    elif db_name == "lancedb":
        retriever_kwargs = {
            "path": str(CONFIG["db_directory"]["lancedb"]),
            "table_name": CONFIG["collection_name"],
            "embedding_model": CONFIG["embedding_model"],
        }

    chain = RAGChain(
        retriever_type=db_name,
        retriever_kwargs=retriever_kwargs
    )

    result = chain.run("What is money laundering?")
    print("Answer:", result["answer"])
    print("\nTop chunk:", result["retrieved_chunks"][0]["text"][:500])

