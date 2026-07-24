from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol, Sequence

from atlas.indexing.models import SearchResult

DEFAULT_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

_STOPWORDS = {
    "a", "al", "algo", "como", "con", "cual", "cuando", "cuanto", "de",
    "del", "el", "ella", "en", "es", "esta", "este", "hay", "la", "las",
    "lo", "los", "me", "mi", "para", "por", "que", "se", "si", "su",
    "sus", "tengo", "un", "una", "y",
}


@dataclass(slots=True)
class RerankedCandidate:
    result: SearchResult
    rerank_score: float
    lexical_score: float | None = None


class Reranker(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
    ) -> list[RerankedCandidate]: ...


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _fold(text))
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lexical_relevance(query: str, document: str) -> float:
    query_tokens = _tokens(query)
    document_tokens = _tokens(document)
    if not query_tokens or not document_tokens:
        return 0.0

    query_set = set(query_tokens)
    document_set = set(document_tokens)
    coverage = len(query_set & document_set) / len(query_set)

    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    document_bigrams = set(zip(document_tokens, document_tokens[1:]))
    bigram_score = (
        len(query_bigrams & document_bigrams) / len(query_bigrams)
        if query_bigrams
        else 0.0
    )

    query_numbers = {token for token in query_set if token.isdigit()}
    number_score = (
        len(query_numbers & document_set) / len(query_numbers)
        if query_numbers
        else 0.0
    )
    return _bounded(0.72 * coverage + 0.18 * bigram_score + 0.10 * number_score)


class HybridLexicalReranker:
    """Low-cost second stage combining semantic and lexical evidence.

    It is deterministic, multilingual at token level and downloads no extra model.
    """

    def __init__(self, *, vector_weight: float = 0.55) -> None:
        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight debe estar entre 0 y 1.")
        self._vector_weight = vector_weight

    @property
    def provider_name(self) -> str:
        return "hybrid"

    @property
    def model_name(self) -> str:
        return "atlas-vector-lexical-reranker-v1"

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
    ) -> list[RerankedCandidate]:
        reranked: list[RerankedCandidate] = []
        for candidate in candidates:
            vector_score = _bounded(candidate.score or 0.0)
            lexical_score = _lexical_relevance(query, candidate.text)
            final_score = (
                self._vector_weight * vector_score
                + (1.0 - self._vector_weight) * lexical_score
            )
            reranked.append(
                RerankedCandidate(
                    result=candidate,
                    rerank_score=_bounded(final_score),
                    lexical_score=lexical_score,
                )
            )
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked


class VectorScoreReranker:
    """No-op reranker useful as a baseline for evaluation."""

    @property
    def provider_name(self) -> str:
        return "vector"

    @property
    def model_name(self) -> str:
        return "original-vector-order"

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
    ) -> list[RerankedCandidate]:
        del query
        reranked = [
            RerankedCandidate(
                result=candidate,
                rerank_score=_bounded(candidate.score or 0.0),
            )
            for candidate in candidates
        ]
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked


class FastEmbedCrossEncoderReranker:
    """Optional model-based reranker executed locally with ONNX."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "FastEmbed no está instalado. Ejecuta: pip install -r requirements.txt"
            ) from error

        self._model_name = model_name
        self._model = TextCrossEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
        )

    @property
    def provider_name(self) -> str:
        return "fastembed-cross-encoder"

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
    ) -> list[RerankedCandidate]:
        if not candidates:
            return []
        raw_scores = list(self._model.rerank(query, [item.text for item in candidates]))
        if len(raw_scores) != len(candidates):
            raise RuntimeError("El reranker devolvió una cantidad incorrecta de scores.")

        reranked: list[RerankedCandidate] = []
        for candidate, raw_score in zip(candidates, raw_scores):
            # Cross-encoder outputs are logits; sigmoid maps them to a readable 0..1 range.
            model_score = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, raw_score))))
            vector_score = _bounded(candidate.score or 0.0)
            final_score = 0.85 * model_score + 0.15 * vector_score
            reranked.append(
                RerankedCandidate(
                    result=candidate,
                    rerank_score=_bounded(final_score),
                )
            )
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked


def create_reranker(
    provider: str,
    *,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    cache_dir: str | None = None,
) -> Reranker:
    normalized = provider.strip().casefold()
    if normalized == "hybrid":
        return HybridLexicalReranker()
    if normalized in {"none", "vector"}:
        return VectorScoreReranker()
    if normalized == "fastembed":
        return FastEmbedCrossEncoderReranker(
            model_name=model_name,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Reranker no compatible: {provider}")
