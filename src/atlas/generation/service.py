from __future__ import annotations

from atlas.rag.models import RetrievalConfig, RetrievalFilters
from atlas.rag.retrieval import RAGRetriever

from .models import AnswerReference, AnswerResult, GenerationConfig
from .prompts import SYSTEM_PROMPT, build_answer_prompt, build_repair_prompt
from .providers import ChatProvider
from .validation import validate_answer

_NO_EVIDENCE = "No encontré información suficiente en los documentos disponibles."
_VALIDATION_FAILED = (
    "No pude validar una respuesta confiable con los documentos disponibles. "
    "Reformula la pregunta o consulta al área responsable."
)


def _reference_from_chunk(chunk: object) -> AnswerReference:
    metadata = getattr(chunk, "metadata")
    citation = getattr(chunk, "citation")
    rank = int(getattr(chunk, "rank"))
    location = citation.split(", ", 1)[1].rstrip("]") if ", " in citation else ""
    return AnswerReference(
        rank=rank,
        citation=citation,
        source_file=str(metadata.get("source_file", "archivo desconocido")),
        location=location,
        category=metadata.get("category"),
        version=metadata.get("version"),
    )


def _best_evidence_score(retrieval: object) -> float:
    chunks = getattr(retrieval, "chunks")
    return max((float(chunk.rerank_score) for chunk in chunks), default=0.0)


class AnswerService:
    """Retrieve evidence, generate an answer, validate it and fail closed."""

    def __init__(self, *, retriever: RAGRetriever, provider: ChatProvider) -> None:
        self._retriever = retriever
        self._provider = provider

    def answer(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        retrieval_config: RetrievalConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> AnswerResult:
        selected_generation = generation_config or GenerationConfig()
        selected_generation.validate()
        retrieval = self._retriever.retrieve(
            query,
            filters=filters,
            config=retrieval_config,
        )
        evidence_score = _best_evidence_score(retrieval)
        base_warnings = list(retrieval.warnings)

        if not retrieval.has_evidence or evidence_score < selected_generation.min_evidence_score:
            if retrieval.has_evidence:
                base_warnings.append(
                    "La evidencia recuperada no superó el umbral mínimo de generación."
                )
            return AnswerResult(
                query=query.strip(),
                answer=_NO_EVIDENCE,
                status="no_evidence",
                retrieval=retrieval.to_dict(),
                provider="none",
                model="none",
                evidence_score=evidence_score,
                warnings=base_warnings,
            )

        prompt = build_answer_prompt(query, retrieval)
        attempts = 0
        last_output = None
        last_validation = None
        for attempt in range(selected_generation.max_validation_retries + 1):
            attempts += 1
            last_output = self._provider.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                config=selected_generation,
            )
            last_validation = validate_answer(
                last_output.text,
                retrieval,
                require_inline_citations=selected_generation.require_inline_citations,
            )
            if last_validation.is_valid:
                references = [
                    _reference_from_chunk(chunk)
                    for chunk in retrieval.chunks
                    if chunk.rank in last_validation.used_source_ranks
                ]
                return AnswerResult(
                    query=query.strip(),
                    answer=last_output.text.strip(),
                    status="answered",
                    references=references,
                    validation=last_validation,
                    retrieval=retrieval.to_dict(),
                    provider=last_output.provider,
                    model=last_output.model,
                    evidence_score=evidence_score,
                    attempts=attempts,
                    warnings=base_warnings + last_validation.warnings,
                    usage=last_output.usage,
                )
            if attempt < selected_generation.max_validation_retries:
                prompt = build_repair_prompt(
                    query,
                    retrieval,
                    last_output.text,
                    last_validation.errors,
                )

        assert last_output is not None and last_validation is not None
        return AnswerResult(
            query=query.strip(),
            answer=_VALIDATION_FAILED,
            status="validation_failed",
            validation=last_validation,
            retrieval=retrieval.to_dict(),
            provider=last_output.provider,
            model=last_output.model,
            evidence_score=evidence_score,
            attempts=attempts,
            warnings=base_warnings + last_validation.errors + last_validation.warnings,
            usage=last_output.usage,
        )
