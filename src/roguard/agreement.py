"""Compare two content-free D1/S1 declarations without copying their values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cli import assess_json

EVIDENCE_FIELDS = {
    "disclosure": ("source_role", "safety_or_support_anchor",
                   "indirect_or_repeated_support_pattern", "explicit_support_request"),
    "support": ("applicable_fields", "passed_fields"),
}
CATEGORY = {"disclosure": "D1", "support": "S1"}


def _validate(payload: Any) -> dict:
    if not isinstance(payload, dict) or set(payload) - {"language", *EVIDENCE_FIELDS} or not (
        set(payload) & set(EVIDENCE_FIELDS)
    ):
        raise ValueError("Agreement inputs require language and D1/S1 declared evidence only")
    return assess_json(payload)


def compare_evidence(left: Any, right: Any) -> dict:
    """Return agreement or abstention; never return raw declaration values."""
    left_report, right_report = _validate(left), _validate(right)
    if left_report["language"] != right_report["language"]:
        raise ValueError("Agreement inputs must use the same language")
    paths = []
    keys = set(left) | set(right)
    for category in EVIDENCE_FIELDS:
        if category not in keys:
            continue
        if category not in left or category not in right:
            paths.append(f"/{category}")
            continue
        for field in EVIDENCE_FIELDS[category]:
            a, b = left[category][field], right[category][field]
            if category == "support" and a is not None and b is not None:
                different = set(a) != set(b)
            else:
                different = a != b
            if different:
                paths.append(f"/{category}/{field}")
    joint_report = left_report if not paths else None
    status = ("needs_adjudication" if paths else
              "incomplete" if joint_report["incomplete"] else "concordant")
    return {
        "schema_version": 1,
        "language": left_report["language"],
        "taxonomy_status": left_report["taxonomy_status"],
        "checked_categories": [CATEGORY[key] for key in EVIDENCE_FIELDS if key in keys],
        "status": status,
        "disagreement_paths": paths,
        "declarations_match": not paths,
        "reviewer_independence_verified": False,
        "caller_facts_verified": False,
        "language_verified": False,
        "review_suggested": bool(paths) or joint_report["review_suggested"],
        "agreed_report": joint_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two local D1/S1 declarations; no content or external action")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    try:
        left = json.loads(args.left.read_text(encoding="utf-8"))
        right = json.loads(args.right.read_text(encoding="utf-8"))
        result = compare_evidence(left, right)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
