from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:p[aá]gina\s+\d+\s+(?:de|/)\s+\d+|-\s*\d+\s*-)+\s*$",
    flags=re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving meaningful line breaks."""

    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = _ZERO_WIDTH_RE.sub("", normalized)

    cleaned_lines: list[str] = []
    for line in normalized.split("\n"):
        compact = _MULTI_SPACE_RE.sub(" ", line).strip()
        if compact and _PAGE_NUMBER_RE.match(compact):
            continue
        cleaned_lines.append(compact)

    cleaned = "\n".join(cleaned_lines)
    cleaned = _MANY_BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()
