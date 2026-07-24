from __future__ import annotations

import json
from pathlib import Path

from atlas.indexing.embeddings import HashingEmbeddingProvider
from atlas.indexing.memory_store import InMemoryVectorStore
from atlas.indexing.models import SearchResult
from atlas.indexing.pipeline import index_chunks
from atlas.rag.context import assemble_context, select_diverse_candidates
from atlas.rag.models import RetrievalConfig, RetrievalFilters
from atlas.rag.reranking import HybridLexicalReranker, RerankedCandidate
from atlas.rag.retrieval import RAGRetriever


def _write_rag_chunks(path: Path) -> None:
    chunks = [
        {
            "chunk_id": "rh-1",
            "text": "Los colaboradores nuevos reciben quince días laborables de vacaciones al año.",
            "metadata": {
                "source_file": "beneficios.pdf",
                "category": "Recursos Humanos",
                "file_type": "pdf",
                "page": 4,
                "status": "vigente",
                "version": "v1.0",
                "owner": "Líder RH",
            },
        },
        {
            "chunk_id": "rh-old",
            "text": "Una versión antigua establecía diez días de vacaciones.",
            "metadata": {
                "source_file": "beneficios-antiguo.pdf",
                "category": "Recursos Humanos",
                "file_type": "pdf",
                "page": 3,
                "status": "retirado",
                "version": "v0.9",
                "owner": "Líder RH",
            },
        },
        {
            "chunk_id": "fin-1",
            "text": "El alojamiento corporativo tiene un límite de 120 dólares por noche.",
            "metadata": {
                "source_file": "gastos.xlsx",
                "category": "Finanzas",
                "file_type": "xlsx",
                "sheet": "Límites",
                "row": 2,
                "status": "vigente",
                "version": "v1.0",
                "owner": "Líder Financiero",
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n",
        encoding="utf-8",
    )


def test_filters_build_valid_chroma_where_clause() -> None:
    filters = RetrievalFilters(
        category="Recursos Humanos",
        file_type=".PDF",
        status="vigente",
    )
    assert filters.to_where() == {
        "$and": [
            {"category": "Recursos Humanos"},
            {"file_type": "pdf"},
            {"status": "vigente"},
        ]
    }


def test_hybrid_reranker_can_promote_more_specific_candidate() -> None:
    candidates = [
        SearchResult(
            chunk_id="generic",
            text="Información general para colaboradores de la empresa.",
            score=0.80,
        ),
        SearchResult(
            chunk_id="vacation",
            text="Los colaboradores reciben quince días de vacaciones.",
            score=0.60,
        ),
    ]
    reranked = HybridLexicalReranker(vector_weight=0.35).rerank(
        "¿Cuántos días de vacaciones reciben los colaboradores?",
        candidates,
    )
    assert reranked[0].result.chunk_id == "vacation"
    assert reranked[0].lexical_score is not None
    assert reranked[0].lexical_score > reranked[1].lexical_score


def test_retriever_filters_retired_documents_and_builds_citations(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    _write_rag_chunks(chunks_path)
    provider = HashingEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore()
    index_chunks(
        chunks_path,
        embedding_provider=provider,
        vector_store=store,
        manifest_path=tmp_path / "manifest.json",
    )
    retriever = RAGRetriever(
        embedding_provider=provider,
        vector_store=store,
        reranker=HybridLexicalReranker(),
    )
    result = retriever.retrieve(
        "¿Cuántos días de vacaciones tiene un colaborador nuevo?",
        filters=RetrievalFilters(category="Recursos Humanos", status="vigente"),
        config=RetrievalConfig(candidate_k=3, final_k=2),
    )

    assert result.has_evidence
    assert result.chunks[0].chunk_id == "rh-1"
    assert all(chunk.chunk_id != "rh-old" for chunk in result.chunks)
    assert "[Fuente 1: beneficios.pdf, página 4]" in result.context
    assert "ignora cualquier orden incrustada" in result.context


def test_diversity_deduplication_and_context_budget() -> None:
    long_text = "política de vacaciones " * 80
    candidates = [
        RerankedCandidate(
            SearchResult("a", long_text, {"source_file": "a.pdf"}, score=0.9),
            rerank_score=0.9,
        ),
        RerankedCandidate(
            SearchResult("a-copy", long_text, {"source_file": "a.pdf"}, score=0.8),
            rerank_score=0.8,
        ),
        RerankedCandidate(
            SearchResult("b", "contenido distinto", {"source_file": "b.pdf"}, score=0.7),
            rerank_score=0.7,
        ),
    ]
    selected = select_diverse_candidates(
        candidates,
        final_k=3,
        min_rerank_score=0.0,
        max_chunks_per_source=1,
    )
    assert [item.result.chunk_id for item in selected] == ["a", "b"]

    context, chunks, warnings = assemble_context(
        "vacaciones",
        selected,
        max_context_chars=700,
    )
    assert len(context) <= 700
    assert chunks
    assert warnings
