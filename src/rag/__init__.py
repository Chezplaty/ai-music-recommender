"""
Retrieval layer (RAG) for the music recommender.

  ingestion.py   - turns catalog songs into rich text documents and indexes them in ChromaDB
  embeddings.py  - text -> vector (sentence-transformers)
  retrieval.py   - semantic search over the index, returning real catalog Song objects
  pipeline.py    - retrieve candidates, then rank them with the existing scorer

Design rule: retrieval only *finds candidates*; the original weighted scorer in
recommender.py still decides the ranking.
"""
