"""Select development-only binary thresholds; never fit on the test set."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from .review import APPLIES_TO, CATEGORIES, KINDS


def calibrate(records: Iterable[tuple[Mapping[str, float], set[str], str]], *, target_recall: float = 0.9) -> dict[str, float]:
    if (type(target_recall) not in (float, int) or
            not math.isfinite(target_recall) or not 0 < target_recall <= 1):
        raise ValueError("target_recall must be in (0, 1]")
    data = list(records)
    if not data:
        raise ValueError("Development records are empty")
    validated = []
    for index, record in enumerate(data, 1):
        if not isinstance(record, tuple) or len(record) != 3:
            raise ValueError(f"Development record {index} needs scores, labels, and source kind")
        scores, labels, kind = record
        if kind not in KINDS:
            raise ValueError(f"Development record {index} has an invalid source kind")
        if not isinstance(scores, Mapping) or set(scores) != set(CATEGORIES):
            raise ValueError(f"Development record {index} must score every category")
        if any(type(value) not in (float, int) or not math.isfinite(value) or
               not 0 <= value <= 1 for value in scores.values()):
            raise ValueError(f"Development record {index} has invalid probabilities")
        if (not isinstance(labels, (set, frozenset)) or
                any(not isinstance(code, str) or code not in CATEGORIES or
                    kind not in APPLIES_TO[code] for code in labels)):
            raise ValueError(f"Development record {index} has invalid or inapplicable labels")
        validated.append((scores, labels, kind))
    result: dict[str, float] = {}
    for code in CATEGORIES:
        pairs = [(float(scores[code]), code in labels) for scores, labels, kind in validated
                 if kind in APPLIES_TO[code]]
        positives = sum(truth for _, truth in pairs)
        if positives == 0:
            raise ValueError(f"No positive development case for {code}")
        if positives == len(pairs):
            raise ValueError(f"No negative development case for {code}")
        eligible = []
        for threshold in sorted({score for score, _ in pairs}):
            tp = sum(score >= threshold and truth for score, truth in pairs)
            fp = sum(score >= threshold and not truth for score, truth in pairs)
            recall = tp / positives
            if recall >= target_recall:
                eligible.append((tp / (tp + fp), threshold))
        result[code] = max(eligible)[1]
    return result
