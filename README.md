# RAG-EVAL-STACK

An LLM workout designed around implementing a RAG pipeline.

**Thesis**: A complete RAG pipeline with three retrieval backends compared, evaluated with RAGAS, with an LLM-as-judge evaluation layer.

## Background

A Retrieval-Augmented Generation (a.k.a. RAG) system consists of two key components:

* A retrieval model that fetches relevant information from an external knowledge source, which could be a database, search engine, or any other information repository.
* A language model that generates responses based on the retrieved knowledge.

There are several ways to implement RAG, including Graph RAG, Hybrid RAG, and Hierarchical RAG, which we'll discuss at the end of this post.

## Design

In this workout I created a simple RAG. The system will have the following components:

``` bash         
User -> question prompt -> | embedding model | -> embedded prompt -> | search vector database | -> extract knowledge / context -> convert context back into text ->  |----------|
        ---------------                                                                                                                                              | chat bot | -> informed response 
               └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────>  |----------|
```

## References

1. Ragas tutorials: https://docs.ragas.io/en/latest/tutorials/rag/