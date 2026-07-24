from __future__ import annotations

from atlas.indexing.embeddings import EmbeddingProvider
from atlas.indexing.pipeline import search_index
from atlas.indexing.store import VectorStore

from .context import assemble_context, select_diverse_candidates
from .models import RetrievalConfig, RetrievalFilters, RetrievalResult
from .reranking import Reranker


class RAGRetriever:
    """Retrieve, filter, rerank, diversify and assemble LLM-ready evidence."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("La pregunta no puede estar vacía.")

        selected_filters = filters or RetrievalFilters()
        selected_config = config or RetrievalConfig()
        selected_config.validate()

        raw_candidates = search_index(
            clean_query,
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
            top_k=selected_config.candidate_k,
            where=selected_filters.to_where(),
        )
        candidates = [
            item
            for item in raw_candidates
            if float(item.score or 0.0) >= selected_config.min_vector_score
        ]
        reranked = self._reranker.rerank(clean_query, candidates)
        selected = select_diverse_candidates(
            reranked,
            final_k=selected_config.final_k,
            min_rerank_score=selected_config.min_rerank_score,
            max_chunks_per_source=selected_config.max_chunks_per_source,
        )

        context, chunks, warnings = assemble_context(
            clean_query,
            selected,
            max_context_chars=selected_config.max_context_chars,
        )
        if not raw_candidates:
            warnings.append("No se encontraron candidatos para los filtros indicados.")
        elif not chunks:
            warnings.append(
                "Los candidatos recuperados no superaron los umbrales de relevancia."
            )

        return RetrievalResult(
            query=clean_query,
            context=context,
            chunks=chunks,
            filters=selected_filters.to_dict(),
            candidate_count=len(raw_candidates),
            warnings=warnings,
            reranker_provider=self._reranker.provider_name,
            reranker_model=self._reranker.model_name,
        )
