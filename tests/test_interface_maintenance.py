from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.app import AtlasRuntime, RuntimeSettings
from atlas.indexing.embeddings import HashingEmbeddingProvider
from atlas.indexing.memory_store import InMemoryVectorStore
from atlas.indexing.models import IndexRecord
from atlas.monitoring import EventStore, summarize_events
from atlas.rag.reranking import HybridLexicalReranker


class FakeReference:
    source_file = "politica.pdf"


class FakeResult:
    status = "answered"
    provider = "extractive"
    model = "demo"
    evidence_score = 0.8
    warnings: list[str] = []
    references = [FakeReference()]


def test_runtime_settings_resolve_paths_and_validate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ATLAS_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("ATLAS_LOG_QUERY_TEXT", "true")
    settings = RuntimeSettings.from_environment(tmp_path)
    assert settings.db_path == tmp_path / "data/vector-store/chroma"
    assert settings.log_query_text is True
    settings.validate()


def test_event_store_hides_query_by_default(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.jsonl")
    event = store.record_answer(
        session_id="session-1",
        query="¿Cuántos días?",
        result=FakeResult(),
        latency_ms=123.45,
    )
    assert "query" not in event
    assert event["query_hash"]
    assert event["reference_files"] == ["politica.pdf"]


def test_event_store_records_feedback_and_rejects_invalid_rating(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.jsonl")
    feedback = store.record_feedback(
        session_id="session-1",
        answer_event_id="answer-1",
        rating="positive",
    )
    assert feedback["rating"] == "positive"
    with pytest.raises(ValueError):
        store.record_feedback(
            session_id="session-1",
            answer_event_id="answer-1",
            rating="neutral",
        )


def test_quality_metrics_cover_unanswered_feedback_and_latency() -> None:
    metrics = summarize_events(
        [
            {"event_type": "answer", "status": "answered", "latency_ms": 100, "evidence_score": 0.8},
            {"event_type": "answer", "status": "no_evidence", "latency_ms": 300, "evidence_score": 0.1},
            {"event_type": "feedback", "rating": "positive"},
            {"event_type": "feedback", "rating": "negative"},
        ]
    )
    assert metrics.total_questions == 2
    assert metrics.unanswered_rate == 0.5
    assert metrics.negative_feedback_rate == 0.5
    assert metrics.average_latency_ms == 200


def test_runtime_health_and_answer_use_existing_pipeline(tmp_path: Path) -> None:
    provider = HashingEmbeddingProvider()
    store = InMemoryVectorStore(collection_name="atlas_test")
    text = (
        "Los colaboradores nuevos de NovaCommerce reciben quince días laborables "
        "de vacaciones al año."
    )
    store.upsert(
        [
            IndexRecord(
                chunk_id="rh-1",
                text=text,
                metadata={
                    "source_file": "politica-beneficios-laborales.pdf",
                    "file_type": "pdf",
                    "category": "Recursos Humanos",
                    "status": "vigente",
                    "version": "v1.0",
                    "owner": "Líder de Recursos Humanos",
                    "page": 4,
                    "chunk_index": 1,
                },
                embedding=provider.embed_query(text),
            )
        ]
    )
    manifest = tmp_path / "index-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "collection_name": "atlas_test",
                "embedding_provider": provider.provider_name,
                "embedding_model": provider.model_name,
                "indexed_at_utc": "2026-07-24T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    settings = RuntimeSettings(
        project_root=tmp_path,
        db_path=tmp_path / "chroma",
        manifest_path=manifest,
        inventory_path=tmp_path / "inventory.csv",
        embedding_provider="hashing",
        embedding_model=provider.model_name,
        embedding_cache_dir=tmp_path / "cache",
        min_evidence_score=0.0,
        candidate_k=4,
        final_k=1,
    )
    runtime = AtlasRuntime(
        settings,
        embedding_provider=provider,
        vector_store=store,
        reranker=HybridLexicalReranker(),
    )
    assert runtime.health().indexed_chunks == 1
    result = runtime.answer(
        "¿Cuántos días laborables de vacaciones reciben los colaboradores nuevos?",
        category="Recursos Humanos",
    )
    assert result.status == "answered"
    assert "quince días" in result.answer
    assert result.references[0].source_file == "politica-beneficios-laborales.pdf"


def test_event_store_reads_jsonl(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.jsonl")
    store.record_answer(
        session_id="session-1",
        query="pregunta",
        result=FakeResult(),
        latency_ms=50,
        include_query_text=True,
    )
    events = list(store.iter_events())
    assert len(events) == 1
    assert events[0]["query"] == "pregunta"
