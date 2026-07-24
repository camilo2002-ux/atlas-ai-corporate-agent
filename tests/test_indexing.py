from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from atlas.indexing.embeddings import HashingEmbeddingProvider
from atlas.indexing.memory_store import InMemoryVectorStore
from atlas.indexing.pipeline import (
    build_where_filter,
    index_chunks,
    iter_chunks_jsonl,
    search_index,
)
from atlas.indexing.store import sanitize_metadata


def _write_chunks(path: Path) -> None:
    chunks = [
        {
            "chunk_id": "rh-1",
            "text": "Los colaboradores reciben quince días laborables de vacaciones.",
            "metadata": {
                "source_file": "beneficios.pdf",
                "category": "Recursos Humanos",
                "file_type": "pdf",
                "page": 4,
            },
        },
        {
            "chunk_id": "fin-1",
            "text": "El límite diario para hotel es de 120 dólares.",
            "metadata": {
                "source_file": "gastos.xlsx",
                "category": "Finanzas",
                "file_type": "xlsx",
                "sheet": "Límites",
            },
        },
        {
            "chunk_id": "tec-1",
            "text": "Un incidente de severidad alta debe escalarse en quince minutos.",
            "metadata": {
                "source_file": "incidentes.md",
                "category": "Tecnología",
                "file_type": "md",
                "section": "Severidad alta",
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    provider = HashingEmbeddingProvider(dimension=64)
    first = provider.embed_query("vacaciones y beneficios")
    second = provider.embed_query("vacaciones y beneficios")
    assert first == second
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0)


def test_jsonl_validation_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"chunk_id":"ok","text":"contenido"}\n{bad json}\n', encoding="utf-8")
    iterator = iter_chunks_jsonl(path)
    assert next(iterator)["chunk_id"] == "ok"
    with pytest.raises(ValueError, match="línea 2"):
        next(iterator)


def test_index_and_search_preserve_metadata(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_chunks(chunks_path)
    provider = HashingEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore()

    report = index_chunks(
        chunks_path,
        embedding_provider=provider,
        vector_store=store,
        reset=True,
        batch_size=2,
        manifest_path=manifest_path,
    )

    assert report.indexed_chunks == 3
    assert store.count() == 3
    assert manifest_path.is_file()

    results = search_index(
        "hotel límite diario 120 dólares",
        embedding_provider=provider,
        vector_store=store,
        top_k=2,
    )
    assert results[0].chunk_id == "fin-1"
    assert results[0].metadata["source_file"] == "gastos.xlsx"
    assert results[0].metadata["embedding_model"] == provider.model_name


def test_metadata_filters_and_sanitization(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    provider = HashingEmbeddingProvider()
    store = InMemoryVectorStore()
    index_chunks(
        chunks_path,
        embedding_provider=provider,
        vector_store=store,
        manifest_path=tmp_path / "manifest.json",
    )

    where = build_where_filter(category="Tecnología", file_type=".MD")
    results = search_index(
        "incidente severidad alta",
        embedding_provider=provider,
        vector_store=store,
        where=where,
    )
    assert [result.chunk_id for result in results] == ["tec-1"]

    metadata = sanitize_metadata({"simple": 3, "nested": {"a": 1}, "empty": None})
    assert metadata["simple"] == 3
    assert metadata["nested"] == '{"a": 1}'
    assert "empty" not in metadata
