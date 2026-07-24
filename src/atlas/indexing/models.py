from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class IndexRecord:
    """A processed chunk paired with its embedding."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


@dataclass(slots=True)
class SearchResult:
    """One vector-search result returned to the RAG layer."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IndexingReport:
    """Summary of an indexing operation."""

    collection_name: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    indexed_chunks: int
    source_jsonl: str
    vector_store_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
