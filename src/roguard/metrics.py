"""Per-category precision and recall on attested abstract cards."""

from __future__ import annotations

from .review import CATEGORIES


def metrics(rows: list[dict], predictions: dict[str, tuple[str, ...]]) -> list[dict]:
    result = []
    for language in sorted({row["language"] for row in rows}):
        subset = [row for row in rows if row["language"] == language]
        for code in CATEGORIES:
            tp = sum(code in row["labels"] and code in predictions[row["id"]] for row in subset)
            fp = sum(code not in row["labels"] and code in predictions[row["id"]] for row in subset)
            fn = sum(code in row["labels"] and code not in predictions[row["id"]] for row in subset)
            result.append({"language": language, "category": code, "support": tp + fn,
                           "precision": tp / (tp + fp) if tp + fp else None,
                           "recall": tp / (tp + fn) if tp + fn else None,
                           "tp": tp, "fp": fp, "fn": fn})
    return result
