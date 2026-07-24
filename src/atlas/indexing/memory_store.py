from __future__ import annotations

import math
from typing import Any, Sequence

from .models import IndexRecord, SearchResult
from .store import sanitize_metadata


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Los embeddings tienen dimensiones diferentes.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(metadata, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_matches(metadata, clause) for clause in where["$or"])
    return all(metadata.get(key) == value for key, value in where.items())


class InMemoryVectorStore:
    """Dependency-free vector store used by automated tests."""

    def __init__(self, collection_name: str = "test_collection") -> None:
        self._collection_name = collection_name
        self._records: dict[str, IndexRecord] = {}

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def count(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        self._records.clear()

    def upsert(self, records: Sequence[IndexRecord]) -> None:
        for record in records:
            self._records[record.chunk_id] = IndexRecord(
                chunk_id=record.chunk_id,
                text=record.text,
                metadata=sanitize_metadata(record.metadata),
                embedding=list(record.embedding),
            )

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        candidates: list[tuple[float, IndexRecord]] = []
        for record in self._records.values():
            if _matches(record.metadata, where):
                candidates.append((_cosine(embedding, record.embedding), record))
        candidates.sort(key=lambda item: item[0], reverse=True)

        return [
            SearchResult(
                chunk_id=record.chunk_id,
                text=record.text,
                metadata=record.metadata,
                distance=1.0 - score,
                score=score,
            )
            for score, record in candidates[:top_k]
        ]
