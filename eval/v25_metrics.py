"""Frozen V25 paired-card metrics for the same-origin synthetic population."""

from __future__ import annotations

import math
from collections import defaultdict

from training.generate_v25_synthetic import D1_FACTORS, S1_FACTORS, _style

THRESHOLD_GRID = tuple(round(index / 20, 2) for index in range(1, 20))
CELLS = ("ro:D1", "ro:S1", "uk:D1", "uk:S1")


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    estimate = successes / total
    correction = z * z / total
    centre = (estimate + correction / 2) / (1 + correction)
    half = z * math.sqrt(estimate * (1 - estimate) / total +
                         z * z / (4 * total * total)) / (1 + correction)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _pair_rows(rows: list[dict], scores: list[float]) -> dict[str, list[tuple]]:
    if len(rows) != len(scores) or len(rows) % 2:
        raise ValueError("V25 rows and scores must align in negative-positive pairs")
    cells = {cell: [] for cell in CELLS}
    for position in range(0, len(rows), 2):
        left, right = rows[position:position + 2]
        negative, positive = scores[position:position + 2]
        for value in (negative, positive):
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("V25 scores must be finite probabilities")
        left_parts, right_parts = left["id"].rsplit("-", 1), right["id"].rsplit("-", 1)
        if (left_parts[0] != right_parts[0] or left_parts[1] != "0" or
                right_parts[1] != "1" or left["labels"] or
                len(right["labels"]) != 1 or
                left["language"] != right["language"] or
                left["source_kind"] != right["source_kind"] or
                left["split"] != right["split"]):
            raise ValueError("V25 pair structure or truth differs")
        code = right["labels"][0]
        cell = f"{right['language']}:{code}"
        if (cell not in cells or
                right["source_kind"] != ("message" if code == "D1" else "response")):
            raise ValueError("V25 language, category, or source kind differs")
        index = int(left_parts[0].split("-")[-1])
        factors = D1_FACTORS if code == "D1" else S1_FACTORS
        factor = factors[index % len(factors)]
        style = _style(left["split"], index, len(factors))
        cells[cell].append((negative, positive, factor, style))
    if any(not items for items in cells.values()):
        raise ValueError("V25 report needs all four bilingual cells")
    return cells


def _cell_report(items: list[tuple], threshold: float) -> dict:
    count = len(items)
    tp = sum(positive >= threshold for _, positive, _, _ in items)
    tn = sum(negative < threshold for negative, _, _, _ in items)
    exact = sum(negative < threshold <= positive for negative, positive, _, _ in items)
    groups: dict[str, list[bool]] = defaultdict(list)
    styles: dict[str, list[bool]] = defaultdict(list)
    for negative, positive, factor, style in items:
        correct = negative < threshold <= positive
        groups[factor].append(correct)
        styles[str(style)].append(correct)
    return {
        "pairs": count, "cards": 2 * count, "exact_pairs": exact,
        "exact_cards": tp + tn, "positive_recovered": tp,
        "negative_correct": tn, "missed_positives": count - tp,
        "false_reviews": count - tn,
        "pair_rate": round(exact / count, 4),
        "pair_rate_95pct_wilson": wilson(exact, count),
        "recall": round(tp / count, 4),
        "specificity": round(tn / count, 4),
        "precision": round(tp / (tp + count - tn), 4) if tp + count - tn else None,
        "by_changed_factor": {key: {"exact_pairs": sum(values), "pairs": len(values)}
                              for key, values in sorted(groups.items())},
        "by_wording_style": {key: {"exact_pairs": sum(values), "pairs": len(values)}
                             for key, values in sorted(styles.items())},
    }


def evaluate(rows: list[dict], scores: list[float], thresholds: dict[str, float]) -> dict:
    if set(thresholds) != set(CELLS) or any(
            type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1
            for value in thresholds.values()):
        raise ValueError("V25 needs a valid fixed cutoff for each language/category cell")
    cells = _pair_rows(rows, scores)
    reports = {cell: _cell_report(cells[cell], thresholds[cell]) for cell in CELLS}
    total = sum(item["pairs"] for item in reports.values())
    exact = sum(item["exact_pairs"] for item in reports.values())
    return {
        "input_scope": "invented_abstract_metadata_only",
        "strict_parse_coverage": 1.0,
        "thresholds": thresholds,
        "pairs": total, "exact_pairs": exact,
        "pair_rate": round(exact / total, 4),
        "pair_rate_95pct_wilson": wilson(exact, total),
        "cells": reports,
    }


def select_thresholds(rows: list[dict], scores: list[float]) -> dict[str, float]:
    """Choose one grid cutoff per cell on development only."""
    cells = _pair_rows(rows, scores)
    result = {}
    for cell, items in cells.items():
        choices = []
        for threshold in THRESHOLD_GRID:
            report = _cell_report(items, threshold)
            choices.append((report["exact_pairs"], report["exact_cards"],
                            -report["false_reviews"], -abs(threshold - 0.5),
                            -threshold, threshold))
        result[cell] = max(choices)[-1]
    return result


def target_passed(report: dict) -> bool:
    if report["strict_parse_coverage"] != 1.0:
        return False
    for cell, item in report["cells"].items():
        if (item["pairs"] != 48 or item["exact_pairs"] < 44 or
                item["positive_recovered"] < 44 or item["negative_correct"] < 44):
            return False
        floor = 9 if cell.endswith(":D1") else 20
        expected = 12 if cell.endswith(":D1") else 24
        if any(entry["pairs"] != expected or entry["exact_pairs"] < floor
               for entry in item["by_changed_factor"].values()):
            return False
    return True
