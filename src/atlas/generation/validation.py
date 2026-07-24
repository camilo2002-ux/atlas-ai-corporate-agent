from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from atlas.rag.models import RetrievalResult

from .models import ValidationResult

_CITATION_PATTERN = re.compile(r"\[Fuente\s+(\d+)\]", flags=re.IGNORECASE)
_FALLBACK = "No encontré información suficiente en los documentos disponibles."

_STOPWORDS = {
    "a", "al", "algo", "como", "con", "cual", "cuando", "cuanto", "de",
    "del", "el", "ella", "en", "es", "esta", "este", "fue", "ha", "hay",
    "la", "las", "lo", "los", "me", "mi", "para", "por", "que", "se",
    "si", "sin", "su", "sus", "tiene", "un", "una", "y",
}

_NUMBER_WORDS = {
    "cero": "0", "uno": "1", "una": "1", "dos": "2", "tres": "3",
    "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
    "nueve": "9", "diez": "10", "once": "11", "doce": "12", "trece": "13",
    "catorce": "14", "quince": "15", "dieciseis": "16", "diecisiete": "17",
    "dieciocho": "18", "diecinueve": "19", "veinte": "20", "veintiuno": "21",
    "veintidos": "22", "veintitres": "23", "veinticuatro": "24",
    "veinticinco": "25", "veintiseis": "26", "veintisiete": "27",
    "veintiocho": "28", "veintinueve": "29", "treinta": "30", "cien": "100",
    "ciento": "100", "mil": "1000",
}


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    )


def _content_tokens(text: str) -> set[str]:
    without_citations = _CITATION_PATTERN.sub("", text)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _fold(without_citations))
        if len(token) > 2 and token not in _STOPWORDS and token not in _NUMBER_WORDS
    }


def _numeric_concepts(text: str) -> set[str]:
    folded = _fold(text)
    concepts = set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?:\s*%)?", folded))
    for token in re.findall(r"[a-z]+", folded):
        if token in _NUMBER_WORDS:
            concepts.add(_NUMBER_WORDS[token])
    return {item.replace(",", ".").replace(" ", "") for item in concepts}


def _sentences(answer: str) -> list[str]:
    lines: list[str] = []
    for line in answer.splitlines():
        clean = line.strip().lstrip("-*• ").strip()
        if not clean or clean.casefold().startswith("fuentes"):
            continue
        lines.extend(
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+(?!\[Fuente\s+\d+\])", clean)
            if item.strip()
        )
    return lines


def _source_text(ranks: Iterable[int], retrieval: RetrievalResult) -> str:
    selected = [chunk.text for chunk in retrieval.chunks if chunk.rank in set(ranks)]
    return " ".join(selected)


def validate_answer(
    answer: str,
    retrieval: RetrievalResult,
    *,
    require_inline_citations: bool = True,
) -> ValidationResult:
    clean_answer = answer.strip()
    errors: list[str] = []
    warnings: list[str] = []
    available_ranks = {chunk.rank for chunk in retrieval.chunks}
    cited_ranks = [int(value) for value in _CITATION_PATTERN.findall(clean_answer)]
    used_ranks = sorted(set(cited_ranks))

    if clean_answer == _FALLBACK:
        if retrieval.has_evidence:
            warnings.append("El modelo usó el fallback aunque existía evidencia recuperada.")
        return ValidationResult(
            is_valid=not retrieval.has_evidence,
            score=1.0 if not retrieval.has_evidence else 0.35,
            used_source_ranks=[],
            errors=[] if not retrieval.has_evidence else [
                "La respuesta descartó evidencia disponible sin justificación."
            ],
            warnings=warnings,
        )

    if not clean_answer:
        errors.append("La respuesta está vacía.")
    if not retrieval.has_evidence:
        errors.append("Se generó una respuesta factual sin evidencia recuperada.")
    if require_inline_citations and not cited_ranks:
        errors.append("La respuesta no contiene citas inline [Fuente N].")

    invalid = sorted(set(cited_ranks) - available_ranks)
    if invalid:
        errors.append(
            "La respuesta cita fuentes inexistentes: "
            + ", ".join(str(rank) for rank in invalid)
            + "."
        )

    supported_sentence_count = 0
    factual_sentence_count = 0
    for sentence in _sentences(clean_answer):
        sentence_without_citation = _CITATION_PATTERN.sub("", sentence).strip()
        # Very short conversational fragments are not treated as factual claims.
        if len(_content_tokens(sentence_without_citation)) < 2 and not _numeric_concepts(
            sentence_without_citation
        ):
            continue
        factual_sentence_count += 1
        ranks = [int(value) for value in _CITATION_PATTERN.findall(sentence)]
        valid_ranks = [rank for rank in ranks if rank in available_ranks]
        if require_inline_citations and not valid_ranks:
            errors.append(f"Afirmación sin cita: {sentence_without_citation[:120]}")
            continue

        evidence = _source_text(valid_ranks, retrieval)
        claim_numbers = _numeric_concepts(sentence_without_citation)
        evidence_numbers = _numeric_concepts(evidence)
        missing_numbers = sorted(claim_numbers - evidence_numbers)
        if missing_numbers:
            errors.append(
                "La afirmación contiene cifras no respaldadas "
                f"({', '.join(missing_numbers)}): {sentence_without_citation[:100]}"
            )
            continue

        claim_tokens = _content_tokens(sentence_without_citation)
        evidence_tokens = _content_tokens(evidence)
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        if claim_tokens and overlap < 0.18:
            errors.append(
                "La afirmación tiene poco respaldo textual en su fuente: "
                f"{sentence_without_citation[:120]}"
            )
            continue
        supported_sentence_count += 1

    sentence_score = (
        supported_sentence_count / factual_sentence_count
        if factual_sentence_count
        else 0.0
    )
    citation_score = 1.0 if cited_ranks and not invalid else 0.0
    score = max(0.0, min(1.0, 0.7 * sentence_score + 0.3 * citation_score))
    return ValidationResult(
        is_valid=not errors,
        score=score,
        used_source_ranks=used_ranks,
        errors=errors,
        warnings=warnings,
    )
