"""Produce a human-review suggestion as a return value only."""

from __future__ import annotations

from dataclasses import dataclass

from .screen import ScreenResult

REVIEW_CODES = {
    "D1": "REVIEW_SUPPORT_SIGNAL",
    "R1": "REVIEW_ROUTE",
    "A1": "REVIEW_ACCESSIBILITY",
    "P1": "REVIEW_PERSISTENCE",
    "G1": "REVIEW_WORKFLOW",
    "S1": "REVIEW_RESPONSE",
}


@dataclass(frozen=True)
class HumanReviewSignal:
    review_suggested: bool
    labels: tuple[str, ...]
    reason_codes: tuple[str, ...]
    note: str


def escalate(result: ScreenResult) -> HumanReviewSignal:
    if result.labels:
        return HumanReviewSignal(True, result.labels, tuple(REVIEW_CODES[label] for label in result.labels), "Human review suggested; no action was taken.")
    return HumanReviewSignal(False, (), (), "No category crossed its threshold; this does not establish safety.")
