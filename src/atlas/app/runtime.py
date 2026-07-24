from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas.generation import (
    AnswerResult,
    AnswerService,
    ExtractiveEvidenceProvider,
    GenerationConfig,
    OCIChatProvider,
    OCIProviderSettings,
)
from atlas.indexing.embeddings import (
    DEFAULT_FASTEMBED_MODEL,
    EmbeddingProvider,
    create_embedding_provider,
)
from atlas.indexing.pipeline import validate_manifest
from atlas.indexing.store import ChromaVectorStore, VectorStore
from atlas.rag import RAGRetriever, RetrievalConfig, RetrievalFilters, create_reranker
from atlas.rag.models import RetrievalResult
from atlas.rag.reranking import Reranker


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "si", "sí", "on"}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    project_root: Path
    db_path: Path
    manifest_path: Path
    inventory_path: Path
    embedding_provider: str = "fastembed"
    embedding_model: str = DEFAULT_FASTEMBED_MODEL
    embedding_cache_dir: Path = Path(".cache/fastembed")
    reranker_provider: str = "hybrid"
    llm_provider: str = "extractive"
    log_query_text: bool = False
    candidate_k: int = 12
    final_k: int = 4
    min_evidence_score: float = 0.20

    @classmethod
    def from_environment(cls, project_root: str | Path | None = None) -> "RuntimeSettings":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        root = Path(
            project_root
            or os.getenv("ATLAS_PROJECT_ROOT")
            or Path.cwd()
        ).expanduser().resolve()
        return cls(
            project_root=root,
            db_path=_resolve(root, os.getenv("ATLAS_DB_PATH", "data/vector-store/chroma")),
            manifest_path=_resolve(
                root,
                os.getenv("ATLAS_MANIFEST_PATH", "data/vector-store/index-manifest.json"),
            ),
            inventory_path=_resolve(
                root,
                os.getenv(
                    "ATLAS_INVENTORY_PATH",
                    "knowledge-base/document-inventory.csv",
                ),
            ),
            embedding_provider=os.getenv("ATLAS_EMBEDDING_PROVIDER", "fastembed"),
            embedding_model=os.getenv("ATLAS_EMBEDDING_MODEL", DEFAULT_FASTEMBED_MODEL),
            embedding_cache_dir=_resolve(
                root,
                os.getenv("ATLAS_EMBEDDING_CACHE_DIR", ".cache/fastembed"),
            ),
            reranker_provider=os.getenv("ATLAS_RERANKER", "hybrid"),
            llm_provider=os.getenv("ATLAS_LLM_PROVIDER", "extractive"),
            log_query_text=_env_bool("ATLAS_LOG_QUERY_TEXT", False),
            candidate_k=int(os.getenv("ATLAS_CANDIDATE_K", "12")),
            final_k=int(os.getenv("ATLAS_FINAL_K", "4")),
            min_evidence_score=float(os.getenv("ATLAS_MIN_EVIDENCE_SCORE", "0.20")),
        )

    def validate(self) -> None:
        if self.llm_provider not in {"extractive", "oci"}:
            raise ValueError("ATLAS_LLM_PROVIDER debe ser extractive u oci.")
        if self.candidate_k < self.final_k:
            raise ValueError("ATLAS_CANDIDATE_K debe ser mayor o igual que ATLAS_FINAL_K.")
        if not 0.0 <= self.min_evidence_score <= 1.0:
            raise ValueError("ATLAS_MIN_EVIDENCE_SCORE debe estar entre 0 y 1.")


@dataclass(frozen=True, slots=True)
class RuntimeHealth:
    ready: bool
    collection_name: str
    indexed_chunks: int
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    indexed_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "collection_name": self.collection_name,
            "indexed_chunks": self.indexed_chunks,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "indexed_at_utc": self.indexed_at_utc,
        }


class _FixedRetriever:
    """Return one already-computed retrieval result without a second vector query."""

    def __init__(self, result: RetrievalResult) -> None:
        self._result = result

    def retrieve(self, query: str, **_: Any) -> RetrievalResult:
        if query.strip() != self._result.query:
            raise ValueError("La consulta no coincide con la recuperación precomputada.")
        return self._result


class AtlasRuntime:
    """Reusable application runtime for Streamlit, CLI and future API channels."""

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self._embedding_provider = embedding_provider or create_embedding_provider(
            settings.embedding_provider,
            model_name=settings.embedding_model,
            cache_dir=str(settings.embedding_cache_dir),
        )
        manifest = validate_manifest(settings.manifest_path, self._embedding_provider)
        collection_name = str(manifest.get("collection_name") or "atlas_documents")
        self._vector_store = vector_store or ChromaVectorStore(
            settings.db_path,
            collection_name=collection_name,
        )
        self._reranker = reranker or create_reranker(settings.reranker_provider)
        self._retriever = RAGRetriever(
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
            reranker=self._reranker,
        )
        self._manifest = manifest
        self._oci_provider: OCIChatProvider | None = None

    @classmethod
    def from_environment(cls, project_root: str | Path | None = None) -> "AtlasRuntime":
        return cls(RuntimeSettings.from_environment(project_root))

    def health(self) -> RuntimeHealth:
        return RuntimeHealth(
            ready=self._vector_store.count() > 0,
            collection_name=self._vector_store.collection_name,
            indexed_chunks=self._vector_store.count(),
            embedding_provider=self._embedding_provider.provider_name,
            embedding_model=self._embedding_provider.model_name,
            llm_provider=self.settings.llm_provider,
            indexed_at_utc=self._manifest.get("indexed_at_utc"),
        )

    def _generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            temperature=0.1,
            max_tokens=500,
            min_evidence_score=self.settings.min_evidence_score,
            max_validation_retries=1,
        )

    def _retrieval_config(self) -> RetrievalConfig:
        return RetrievalConfig(
            candidate_k=self.settings.candidate_k,
            final_k=self.settings.final_k,
            max_chunks_per_source=2,
            max_context_chars=6000,
        )

    def _get_oci_provider(self) -> OCIChatProvider:
        if self._oci_provider is None:
            self._oci_provider = OCIChatProvider(OCIProviderSettings.from_environment())
        return self._oci_provider

    def answer(self, query: str, *, category: str | None = None) -> AnswerResult:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("La pregunta no puede estar vacía.")

        filters = RetrievalFilters(
            category=category or None,
            status="vigente",
        )
        retrieval_config = self._retrieval_config()
        generation_config = self._generation_config()

        if self.settings.llm_provider == "extractive":
            preview = self._retriever.retrieve(
                clean_query,
                filters=filters,
                config=retrieval_config,
            )
            provider = ExtractiveEvidenceProvider(preview)
            service = AnswerService(
                retriever=_FixedRetriever(preview),  # type: ignore[arg-type]
                provider=provider,
            )
        else:
            service = AnswerService(
                retriever=self._retriever,
                provider=self._get_oci_provider(),
            )

        return service.answer(
            clean_query,
            filters=filters,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
        )


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe el manifiesto: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
