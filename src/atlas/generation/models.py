from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class GenerationConfig:
    """Controls answer generation and post-generation validation."""

    temperature: float = 0.1
    max_tokens: int = 500
    min_evidence_score: float = 0.20
    max_validation_retries: int = 1
    require_inline_citations: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature debe estar entre 0 y 2.")
        if self.max_tokens < 32:
            raise ValueError("max_tokens debe ser al menos 32.")
        if not 0.0 <= self.min_evidence_score <= 1.0:
            raise ValueError("min_evidence_score debe estar entre 0 y 1.")
        if self.max_validation_retries < 0:
            raise ValueError("max_validation_retries no puede ser negativo.")


@dataclass(slots=True)
class LLMOutput:
    """Raw text and optional usage information returned by a provider."""

    text: str
    provider: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerReference:
    rank: int
    citation: str
    source_file: str
    location: str
    category: str | None = None
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    score: float
    used_source_ranks: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    query: str
    answer: str
    status: str
    references: list[AnswerReference] = field(default_factory=list)
    validation: ValidationResult | None = None
    retrieval: dict[str, Any] = field(default_factory=dict)
    provider: str = "none"
    model: str = "none"
    evidence_score: float = 0.0
    attempts: int = 0
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "status": self.status,
            "references": [reference.to_dict() for reference in self.references],
            "validation": self.validation.to_dict() if self.validation else None,
            "retrieval": self.retrieval,
            "provider": self.provider,
            "model": self.model,
            "evidence_score": self.evidence_score,
            "attempts": self.attempts,
            "warnings": self.warnings,
            "usage": self.usage,
            "answered": self.answered,
        }
