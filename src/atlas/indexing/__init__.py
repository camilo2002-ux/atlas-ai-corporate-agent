"""Embedding generation, vector indexing and semantic retrieval for Atlas."""

from .embeddings import DEFAULT_FASTEMBED_MODEL, create_embedding_provider
from .pipeline import build_where_filter, index_chunks, search_index
from .store import ChromaVectorStore

__all__ = [
    "ChromaVectorStore",
    "DEFAULT_FASTEMBED_MODEL",
    "build_where_filter",
    "create_embedding_provider",
    "index_chunks",
    "search_index",
]
