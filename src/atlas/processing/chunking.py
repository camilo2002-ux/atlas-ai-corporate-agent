from __future__ import annotations

from .cleaning import clean_text
from .models import DocumentUnit


def _find_boundary(text: str, start: int, desired_end: int) -> int:
    """Find a readable boundary near the requested end position."""

    if desired_end >= len(text):
        return len(text)

    search_start = max(start + 1, desired_end - 180)
    window = text[search_start:desired_end]

    for separator in ("\n\n", "\n", ". ", "; ", ", ", " "):
        position = window.rfind(separator)
        if position != -1:
            boundary = search_start + position + len(separator)
            if boundary > start:
                return boundary

    return desired_end


def split_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 180,
) -> list[str]:
    """Split text into overlapping, word-aware character windows."""

    if max_chars < 200:
        raise ValueError("max_chars debe ser al menos 200")
    if overlap_chars < 0:
        raise ValueError("overlap_chars no puede ser negativo")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars debe ser menor que max_chars")

    cleaned = clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0

    while start < len(cleaned):
        desired_end = min(start + max_chars, len(cleaned))
        end = _find_boundary(cleaned, start, desired_end)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(cleaned):
            break

        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end

        while next_start < end and not cleaned[next_start].isspace():
            next_start += 1
        start = min(next_start, len(cleaned))

    return chunks


def chunk_units(
    units: list[DocumentUnit],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 180,
) -> list[DocumentUnit]:
    """Chunk each logical unit without crossing its source boundary."""

    output: list[DocumentUnit] = []
    for unit_index, unit in enumerate(units, start=1):
        pieces = split_text(
            unit.text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        for part_index, piece in enumerate(pieces, start=1):
            metadata = dict(unit.metadata)
            metadata["unit_index"] = unit_index
            metadata["part_index"] = part_index
            metadata["parts_in_unit"] = len(pieces)
            output.append(DocumentUnit(text=piece, metadata=metadata))
    return output
