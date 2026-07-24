from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    total_questions: int = 0
    answered: int = 0
    no_evidence: int = 0
    validation_failed: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    unanswered_rate: float = 0.0
    negative_feedback_rate: float = 0.0
    average_latency_ms: float = 0.0
    average_evidence_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_events(events: Iterable[dict[str, Any]]) -> QualityMetrics:
    answer_events = [event for event in events if event.get("event_type") == "answer"]
    feedback_events = [event for event in events if event.get("event_type") == "feedback"]

    total = len(answer_events)
    answered = sum(event.get("status") == "answered" for event in answer_events)
    no_evidence = sum(event.get("status") == "no_evidence" for event in answer_events)
    validation_failed = sum(
        event.get("status") == "validation_failed" for event in answer_events
    )
    positive = sum(event.get("rating") == "positive" for event in feedback_events)
    negative = sum(event.get("rating") == "negative" for event in feedback_events)
    feedback_total = positive + negative

    latencies = [
        float(event["latency_ms"])
        for event in answer_events
        if isinstance(event.get("latency_ms"), (int, float))
    ]
    evidence_scores = [
        float(event["evidence_score"])
        for event in answer_events
        if isinstance(event.get("evidence_score"), (int, float))
    ]

    return QualityMetrics(
        total_questions=total,
        answered=answered,
        no_evidence=no_evidence,
        validation_failed=validation_failed,
        positive_feedback=positive,
        negative_feedback=negative,
        unanswered_rate=((no_evidence + validation_failed) / total if total else 0.0),
        negative_feedback_rate=(negative / feedback_total if feedback_total else 0.0),
        average_latency_ms=mean(latencies) if latencies else 0.0,
        average_evidence_score=mean(evidence_scores) if evidence_scores else 0.0,
    )
