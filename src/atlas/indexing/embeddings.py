from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence


DEFAULT_FASTEMBED_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class EmbeddingProvider(Protocol):
    """Common interface for document and query embeddings."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(value / norm) for value in vector]


class FastEmbedProvider:
    """Multilingual local embeddings backed by ONNX through FastEmbed."""

    def __init__(
        self,
        model_name: str = DEFAULT_FASTEMBED_MODEL,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "FastEmbed no está instalado. Ejecuta: pip install -r requirements.txt"
            ) from error

        self._model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
        )
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return "fastembed"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed_query("dimensión del embedding")
            self._dimension = len(probe)
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [_normalize(vector.tolist()) for vector in self._model.embed(list(texts))]
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("La consulta para embeddings no puede estar vacía.")
        vector = next(iter(self._model.embed([text])))
        normalized = _normalize(vector.tolist())
        self._dimension = len(normalized)
        return normalized


class HashingEmbeddingProvider:
    """Small deterministic provider for tests and offline smoke checks.

    It is intentionally simple and must not be used as the final semantic model.
    """

    def __init__(self, dimension: int = 128) -> None:
        if dimension < 16:
            raise ValueError("La dimensión debe ser al menos 16.")
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return "hashing"

    @property
    def model_name(self) -> str:
        return f"atlas-hashing-v1-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("La consulta para embeddings no puede estar vacía.")
        return self._embed(text)


def create_embedding_provider(
    provider: str,
    *,
    model_name: str = DEFAULT_FASTEMBED_MODEL,
    cache_dir: str | None = None,
) -> EmbeddingProvider:
    normalized = provider.strip().casefold()
    if normalized == "fastembed":
        return FastEmbedProvider(model_name=model_name, cache_dir=cache_dir)
    if normalized == "hashing":
        return HashingEmbeddingProvider()
    raise ValueError(f"Proveedor de embeddings no compatible: {provider}")
