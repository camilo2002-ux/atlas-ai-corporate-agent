from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .embeddings import EmbeddingProvider
from .models import IndexRecord, IndexingReport, SearchResult
from .store import VectorStore


def iter_chunks_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"No existe el archivo de chunks: {source}")

    with source.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON inválido en {source}, línea {line_number}: {error.msg}"
                ) from error

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            metadata = chunk.get("metadata", {})
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"Falta chunk_id en la línea {line_number}.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Falta texto en la línea {line_number}.")
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata debe ser un objeto en la línea {line_number}.")
            yield {"chunk_id": chunk_id, "text": text, "metadata": metadata}


def _batched(items: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    if size < 1:
        raise ValueError("El tamaño del lote debe ser al menos 1.")
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_where_filter(
    *,
    category: str | None = None,
    file_type: str | None = None,
    source_file: str | None = None,
) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = []
    if category:
        filters.append({"category": category})
    if file_type:
        filters.append({"file_type": file_type.casefold().lstrip(".")})
    if source_file:
        filters.append({"source_file": source_file})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def index_chunks(
    chunks_jsonl: str | Path,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    batch_size: int = 32,
    reset: bool = False,
    manifest_path: str | Path = "data/vector-store/index-manifest.json",
) -> IndexingReport:
    source = Path(chunks_jsonl)
    if reset:
        vector_store.reset()

    indexed = 0
    dimension: int | None = None
    for batch in _batched(iter_chunks_jsonl(source), batch_size):
        texts = [item["text"] for item in batch]
        embeddings = embedding_provider.embed_documents(texts)
        if len(embeddings) != len(batch):
            raise RuntimeError("El proveedor devolvió una cantidad incorrecta de embeddings.")

        records: list[IndexRecord] = []
        for item, embedding in zip(batch, embeddings):
            if dimension is None:
                dimension = len(embedding)
            elif len(embedding) != dimension:
                raise RuntimeError("El proveedor devolvió dimensiones inconsistentes.")
            metadata = {
                **item["metadata"],
                "embedding_provider": embedding_provider.provider_name,
                "embedding_model": embedding_provider.model_name,
            }
            records.append(
                IndexRecord(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    metadata=metadata,
                    embedding=embedding,
                )
            )
        vector_store.upsert(records)
        indexed += len(records)

    if indexed == 0 or dimension is None:
        raise ValueError("El archivo JSONL no contiene chunks utilizables.")

    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_name": vector_store.collection_name,
        "embedding_provider": embedding_provider.provider_name,
        "embedding_model": embedding_provider.model_name,
        "embedding_dimension": dimension,
        "indexed_chunks": indexed,
        "collection_total": vector_store.count(),
        "source_jsonl": str(source),
    }
    manifest.write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return IndexingReport(
        collection_name=vector_store.collection_name,
        embedding_provider=embedding_provider.provider_name,
        embedding_model=embedding_provider.model_name,
        embedding_dimension=dimension,
        indexed_chunks=indexed,
        source_jsonl=str(source),
        vector_store_path=str(getattr(vector_store, "_path", "memory")),
        manifest_path=str(manifest),
    )


def search_index(
    query: str,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("La consulta no puede estar vacía.")
    query_embedding = embedding_provider.embed_query(query)
    return vector_store.query(query_embedding, top_k=top_k, where=where)


def validate_manifest(
    manifest_path: str | Path,
    embedding_provider: EmbeddingProvider,
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el manifiesto {path}. Ejecuta primero la indexación."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_provider = manifest.get("embedding_provider")
    expected_model = manifest.get("embedding_model")
    if expected_provider != embedding_provider.provider_name:
        raise ValueError(
            "El índice fue creado con otro proveedor de embeddings: "
            f"{expected_provider}."
        )
    if expected_model != embedding_provider.model_name:
        raise ValueError(
            "El índice fue creado con otro modelo de embeddings: "
            f"{expected_model}."
        )
    return manifest
