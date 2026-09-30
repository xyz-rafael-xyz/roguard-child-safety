"""Measure two local streams of content-free D1/S1 reviewer declarations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .agreement import CATEGORY, compare_evidence
from .cli import assess_json

ITEM_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _one_category(record: Any) -> tuple[str, dict]:
    if (not isinstance(record, dict) or set(record) != {"item_id", "card"} or
            not isinstance(record["item_id"], str) or
            not ITEM_ID.fullmatch(record["item_id"])):
        raise ValueError("Each annotation row needs one opaque item_id and one card")
    card = record["card"]
    if not isinstance(card, dict) or set(card) not in (
        {"language", "disclosure"}, {"language", "support"}
    ):
        raise ValueError("Each annotation row must contain language and exactly one D1 or S1 card")
    return next(name for name in CATEGORY if name in card), card


def _kappa(matrix: dict[str, int]) -> float | None:
    a, b, c, d = (matrix[key] for key in (
        "pass_pass", "pass_review", "review_pass", "review_review"))
    count = a + b + c + d
    if not count:
        return None
    observed = (a + d) / count
    left_pass = (a + b) / count
    right_pass = (a + c) / count
    expected = left_pass * right_pass + (1 - left_pass) * (1 - right_pass)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def compare_evidence_batches(left: list[Any], right: list[Any]) -> dict:
    """Compute decision agreement on complete pairs without returning declared values."""
    if not isinstance(left, list) or not isinstance(right, list) or not left or len(left) != len(right):
        raise ValueError("Supply nonempty, equally sized annotation streams")
    left_ids = [record.get("item_id") if isinstance(record, dict) else None for record in left]
    right_ids = [record.get("item_id") if isinstance(record, dict) else None for record in right]
    for record in left + right:
        _one_category(record)
    if (len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(right_ids) or
            set(left_ids) != set(right_ids)):
        raise ValueError("Annotation streams need the same distinct opaque item IDs")
    right_by_id = {record["item_id"]: record for record in right}
    language = None
    per_category = {
        code: {"rows": 0, "complete_pairs": 0, "decision_agreements": 0,
               "matrix": {key: 0 for key in (
                   "pass_pass", "pass_review", "review_pass", "review_review")}}
        for code in CATEGORY.values()
    }
    disagreements = []
    incomplete = []
    exact_fact_matches = 0
    for index, left_record in enumerate(left, 1):
        right_record = right_by_id[left_record["item_id"]]
        kind_a, a = _one_category(left_record)
        kind_b, b = _one_category(right_record)
        if kind_a != kind_b:
            raise ValueError(f"Annotation row {index} compares different categories")
        report_a, report_b = assess_json(a), assess_json(b)
        if report_a["language"] != report_b["language"]:
            raise ValueError(f"Annotation row {index} compares different languages")
        if language is None:
            language = report_a["language"]
        elif language != report_a["language"]:
            raise ValueError("An annotation stream must use one language")
        comparison = compare_evidence(a, b)
        code = CATEGORY[kind_a]
        summary = per_category[code]
        summary["rows"] += 1
        if comparison["declarations_match"]:
            exact_fact_matches += 1
        else:
            disagreements.append(index)
        status_a = report_a["findings"][0]["status"]
        status_b = report_b["findings"][0]["status"]
        if "incomplete" in (status_a, status_b):
            incomplete.append(index)
            continue
        summary["complete_pairs"] += 1
        summary["decision_agreements"] += status_a == status_b
        summary["matrix"][f"{status_a}_{status_b}"] += 1
    for summary in per_category.values():
        summary["decision_agreement_rate"] = (
            summary["decision_agreements"] / summary["complete_pairs"]
            if summary["complete_pairs"] else None)
        summary["cohen_kappa"] = _kappa(summary["matrix"])
    return {
        "schema_version": 1,
        "language": language,
        "rows": len(left),
        "opaque_item_ids_matched": True,
        "exact_fact_matches": exact_fact_matches,
        "disagreement_rows": disagreements,
        "incomplete_rows": incomplete,
        "per_category": per_category,
        "reviewer_independence_verified": False,
        "caller_facts_verified": False,
        "language_verified": False,
        "label_validity_verified": False,
    }


def _read_jsonl(path: Path) -> list[Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Annotation stream is empty")
    rows = []
    for index, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"Annotation line {index} is blank")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Annotation line {index} is not JSON") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit two ordered, content-free D1/S1 annotation streams locally")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    try:
        result = compare_evidence_batches(_read_jsonl(args.left), _read_jsonl(args.right))
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
