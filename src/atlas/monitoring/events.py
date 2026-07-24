from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_WRITE_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:16]


class EventStore:
    """Append-only JSONL event store for lightweight quality monitoring."""

    def __init__(self, path: str | Path = "data/feedback/events.jsonl") -> None:
        self.path = Path(path)

    def _append(self, event: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event_id": uuid.uuid4().hex, "timestamp_utc": _utc_now(), **event}
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with _WRITE_LOCK:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        return payload

    def record_answer(
        self,
        *,
        session_id: str,
        query: str,
        result: Any,
        latency_ms: float,
        include_query_text: bool = False,
    ) -> dict[str, Any]:
        references = [
            getattr(reference, "source_file", "")
            for reference in getattr(result, "references", [])
        ]
        event: dict[str, Any] = {
            "event_type": "answer",
            "session_id": session_id,
            "query_hash": _query_hash(query),
            "query_length": len(query),
            "status": getattr(result, "status", "unknown"),
            "provider": getattr(result, "provider", "unknown"),
            "model": getattr(result, "model", "unknown"),
            "evidence_score": float(getattr(result, "evidence_score", 0.0)),
            "latency_ms": round(float(latency_ms), 2),
            "reference_files": references,
            "warning_count": len(getattr(result, "warnings", [])),
        }
        if include_query_text:
            event["query"] = query.strip()
        return self._append(event)

    def record_feedback(
        self,
        *,
        session_id: str,
        answer_event_id: str,
        rating: str,
    ) -> dict[str, Any]:
        normalized = rating.strip().casefold()
        if normalized not in {"positive", "negative"}:
            raise ValueError("rating debe ser positive o negative.")
        return self._append(
            {
                "event_type": "feedback",
                "session_id": session_id,
                "answer_event_id": answer_event_id,
                "rating": normalized,
            }
        )

    def iter_events(self) -> Iterable[dict[str, Any]]:
        if not self.path.is_file():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Evento JSON inválido en la línea {line_number}."
                    ) from error
                if isinstance(event, dict):
                    events.append(event)
        return events
