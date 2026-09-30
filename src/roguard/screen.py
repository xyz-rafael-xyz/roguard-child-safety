"""Read-only classifier wrapper; the caller supplies a fitted backend and thresholds."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol

from .review import ACTIVE_LANGUAGES, APPLIES_TO, CATEGORIES, KINDS


class ScoreBackend(Protocol):
    def score(self, text: str, language: str, source_kind: str) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class ScreenResult:
    language: str
    source_kind: str
    scores: dict[str, float]
    labels: tuple[str, ...]


def screen(
    text: str, *, language: str, source_kind: str, backend: ScoreBackend,
    thresholds: Mapping[str, float],
) -> ScreenResult:
    if (language not in ACTIVE_LANGUAGES or source_kind not in KINDS or
            not isinstance(text, str) or not text.strip()):
        raise ValueError("Specify nonempty text, an active taxonomy language, and a supported source kind")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(CATEGORIES):
        raise ValueError("A calibrated threshold is required for every category")
    for name, value in thresholds.items():
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid score or threshold for {name}")
    raw = dict(backend.score(text, language, source_kind))
    if set(raw) != set(CATEGORIES):
        raise ValueError("Backend must score every category")
    for name, value in raw.items():
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid score or threshold for {name}")
    labels = tuple(code for code in CATEGORIES if source_kind in APPLIES_TO[code] and raw[code] >= thresholds[code])
    return ScreenResult(language, source_kind, raw, labels)
