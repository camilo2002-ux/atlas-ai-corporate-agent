from __future__ import annotations

from dataclasses import dataclass

from atlas.generation import (
    AnswerService,
    ExtractiveEvidenceProvider,
    GenerationConfig,
    StaticChatProvider,
    validate_answer,
)
from atlas.generation.prompts import SYSTEM_PROMPT, build_answer_prompt
from atlas.rag.models import RetrievalResult, RetrievedChunk


def _retrieval(*, with_evidence: bool = True) -> RetrievalResult:
    chunks = []
    if with_evidence:
        chunks = [
            RetrievedChunk(
                rank=1,
                chunk_id="rh-1",
                text=(
                    "Los colaboradores nuevos de NovaCommerce reciben quince días "
                    "laborables de vacaciones al año. Las solicitudes deben registrarse "
                    "con al menos diez días de anticipación."
                ),
                metadata={
                    "source_file": "beneficios.pdf",
                    "category": "Recursos Humanos",
                    "version": "v1.0",
                    "owner": "Líder RH",
                    "page": 4,
                },
                vector_score=0.60,
                rerank_score=0.44,
                citation="[Fuente 1: beneficios.pdf, página 4]",
                lexical_score=0.24,
            )
        ]
    return RetrievalResult(
        query="¿Cuántos días de vacaciones recibe un colaborador nuevo?",
        context="[Fuente 1: beneficios.pdf, página 4]\nContenido: quince días",
        chunks=chunks,
        candidate_count=len(chunks),
        reranker_provider="hybrid",
        reranker_model="test",
    )


def test_prompt_requires_grounding_and_inline_citations() -> None:
    retrieval = _retrieval()
    prompt = build_answer_prompt(retrieval.query, retrieval)
    assert "exclusivamente" in SYSTEM_PROMPT
    assert "[Fuente N]" in SYSTEM_PROMPT
    assert retrieval.context in prompt


def test_validator_accepts_grounded_answer_and_word_number_equivalence() -> None:
    result = validate_answer(
        "Los colaboradores nuevos reciben 15 días laborables de vacaciones al año. [Fuente 1]",
        _retrieval(),
    )
    assert result.is_valid
    assert result.used_source_ranks == [1]


def test_validator_rejects_invented_source_and_number() -> None:
    result = validate_answer(
        "Los colaboradores reciben treinta días de vacaciones. [Fuente 9]",
        _retrieval(),
    )
    assert not result.is_valid
    assert any("inexistentes" in error for error in result.errors)


def test_extractive_provider_builds_cited_answer() -> None:
    retrieval = _retrieval()
    provider = ExtractiveEvidenceProvider(retrieval)
    output = provider.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_answer_prompt(retrieval.query, retrieval),
        config=GenerationConfig(),
    )
    assert "quince días" in output.text
    assert "[Fuente 1]" in output.text
    assert validate_answer(output.text, retrieval).is_valid


@dataclass
class _StubRetriever:
    result: RetrievalResult

    def retrieve(self, query: str, **_: object) -> RetrievalResult:
        assert query
        return self.result


def test_service_falls_back_without_calling_model() -> None:
    provider = StaticChatProvider(["Respuesta inventada. [Fuente 1]"])
    service = AnswerService(
        retriever=_StubRetriever(_retrieval(with_evidence=False)),  # type: ignore[arg-type]
        provider=provider,
    )
    result = service.answer("¿Cuál es la misión a Marte?")
    assert result.status == "no_evidence"
    assert provider.calls == 0
    assert "No encontré información suficiente" in result.answer


def test_service_repairs_invalid_answer_before_returning() -> None:
    provider = StaticChatProvider(
        [
            "Los colaboradores reciben treinta días. [Fuente 1]",
            "Los colaboradores nuevos reciben quince días laborables de vacaciones al año. [Fuente 1]",
        ]
    )
    service = AnswerService(
        retriever=_StubRetriever(_retrieval()),  # type: ignore[arg-type]
        provider=provider,
    )
    result = service.answer(
        _retrieval().query,
        generation_config=GenerationConfig(max_validation_retries=1),
    )
    assert result.status == "answered"
    assert result.attempts == 2
    assert provider.calls == 2
    assert result.validation and result.validation.is_valid
    assert result.references[0].source_file == "beneficios.pdf"
