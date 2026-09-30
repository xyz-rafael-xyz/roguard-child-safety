"""V26 paired metrics with explicit registered wording styles and factors."""

from __future__ import annotations

from collections import defaultdict

from eval.v25_metrics import CELLS, select_thresholds, target_passed
from eval.v25_metrics import evaluate as _base_evaluate


def evaluate(rows: list[dict], scores: list[float], thresholds: dict[str, float]) -> dict:
    report = _base_evaluate(rows, scores, thresholds)
    groups = {cell: {"factor": defaultdict(list), "style": defaultdict(list)}
              for cell in CELLS}
    for index in range(0, len(rows), 2):
        left, right = rows[index:index + 2]
        if (left["changed_factor"] != right["changed_factor"] or
                left["style_index"] != right["style_index"] or
                left["language"] != right["language"]):
            raise ValueError("V26 pair metadata differs between variants")
        code = right["labels"][0]
        cell = f"{right['language']}:{code}"
        correct = scores[index] < thresholds[cell] <= scores[index + 1]
        groups[cell]["factor"][left["changed_factor"]].append(correct)
        groups[cell]["style"][str(left["style_index"])].append(correct)
    for cell in CELLS:
        for key, output_key in (("factor", "by_changed_factor"),
                                ("style", "by_wording_style")):
            report["cells"][cell][output_key] = {
                name: {"exact_pairs": sum(values), "pairs": len(values)}
                for name, values in sorted(groups[cell][key].items())
            }
    return report
