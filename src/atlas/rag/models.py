from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalFilters:
    """Metadata restrictions applied before vector similarity search."""

    category: str | None = None
    file_type: str | None = None
    source_file: str | None = None
    status: str | None = None
    version: str | None = None
    owner: str | None = None

    def to_where(self) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        values = {
            "category": self.category,
            "file_type": (
                self.file_type.casefold().lstrip(".") if self.file_type else None
            ),
            "source_file": self.source_file,
            "status": self.status,
            "version": self.version,
            "owner": self.owner,
        }
        for key, value in values.items():
            if value:
                clauses.append({key: value})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(slots=True)
class RetrievalConfig:
    """Controls broad retrieval, reranking and final context size."""

    candidate_k: int = 12
    final_k: int = 4
    min_vector_score: float = 0.0
    min_rerank_score: float = 0.0
    max_chunks_per_source: int = 2
    max_context_chars: int = 6000

    def validate(self) -> None:
        if self.candidate_k < 1:
            raise ValueError("candidate_k debe ser al menos 1.")
        if self.final_k < 1:
            raise ValueError("final_k debe ser al menos 1.")
        if self.candidate_k < self.final_k:
            raise ValueError("candidate_k debe ser mayor o igual que final_k.")
        if self.max_chunks_per_source < 1:
            raise ValueError("max_chunks_per_source debe ser al menos 1.")
        if self.max_context_chars < 500:
            raise ValueError("max_context_chars debe ser al menos 500.")
        for name, value in (
            ("min_vector_score", self.min_vector_score),
            ("min_rerank_score", self.min_rerank_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} debe estar entre 0 y 1.")


@dataclass(slots=True)
class RetrievedChunk:
    """Candidate selected for the final RAG context."""

    rank: int
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float
    rerank_score: float
    citation: str
    lexical_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    """Complete output of the retrieval layer, ready for an LLM."""

    query: str
    context: str
    chunks: list[RetrievedChunk]
    filters: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0
    warnings: list[str] = field(default_factory=list)
    reranker_provider: str = "none"
    reranker_model: str = "none"

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "context": self.context,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "filters": self.filters,
            "candidate_count": self.candidate_count,
            "warnings": self.warnings,
            "reranker_provider": self.reranker_provider,
            "reranker_model": self.reranker_model,
            "has_evidence": self.has_evidence,
        }
