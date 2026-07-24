from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from .models import IndexRecord, SearchResult


class VectorStore(Protocol):
    @property
    def collection_name(self) -> str: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...

    def upsert(self, records: Sequence[IndexRecord]) -> None: ...

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert metadata to scalar values accepted by vector stores."""

    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean_key = str(key)
        if isinstance(value, (str, int, float, bool)):
            sanitized[clean_key] = value
        else:
            sanitized[clean_key] = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
    return sanitized


class ChromaVectorStore:
    """Persistent Chroma collection using externally generated embeddings."""

    def __init__(
        self,
        path: str | Path,
        *,
        collection_name: str = "atlas_documents",
    ) -> None:
        try:
            import chromadb
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "Chroma no está instalado. Ejecuta: pip install -r requirements.txt"
            ) from error

        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self._path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
        )

    def upsert(self, records: Sequence[IndexRecord]) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[record.chunk_id for record in records],
            embeddings=[record.embedding for record in records],
            documents=[record.text for record in records],
            metadatas=[sanitize_metadata(record.metadata) for record in records],
        )

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k debe ser al menos 1.")
        available = self.count()
        if available == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [list(embedding)],
            "n_results": min(top_k, available),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else None
            score = None if distance is None else 1.0 / (1.0 + max(distance, 0.0))
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=documents[index] if index < len(documents) else "",
                    metadata=metadatas[index] if index < len(metadatas) else {},
                    distance=distance,
                    score=score,
                )
            )
        return results
