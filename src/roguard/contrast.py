"""One-fact, read-only contrasts for caller-declared fictional contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cli import assess_json


def _changed_leaves(before: Any, after: Any, path: str = "") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        if before.keys() != after.keys():
            raise ValueError("Contrast inputs must have the same object fields")
        return [leaf for key in before for leaf in _changed_leaves(
            before[key], after[key], path + "/" + key.replace("~", "~0").replace("/", "~1"))]
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            raise ValueError("Contrast arrays must have the same length")
        return [leaf for index in range(len(before)) for leaf in _changed_leaves(
            before[index], after[index], path + f"/{index}")]
    if isinstance(before, (dict, list)) or isinstance(after, (dict, list)):
        raise ValueError("Contrast inputs must keep the same nested shape")
    return [path] if type(before) is not type(after) or before != after else []


def contrast_declared_contracts(before: dict, after: dict) -> dict:
    """Compare exactly one changed scalar; return decisions without input values."""

    first = assess_json(before)
    second = assess_json(after)
    if first["language"] != second["language"]:
        raise ValueError("Contrast inputs must use the same language")
    changed = _changed_leaves(before, after)
    if len(changed) != 1 or changed[0] == "/language":
        raise ValueError("Contrast requires exactly one changed contract value")
    first_findings = {(item["category"], item["source_kind"]): item for item in first["findings"]}
    second_findings = {(item["category"], item["source_kind"]): item for item in second["findings"]}
    if first_findings.keys() != second_findings.keys():
        raise ValueError("Contrast inputs must contain the same contracts")
    transitions = []
    for key, old in first_findings.items():
        new = second_findings[key]
        if old["status"] != new["status"] or old["reason_codes"] != new["reason_codes"]:
            transitions.append({
                "category": old["category"], "source_kind": old["source_kind"],
                "before_status": old["status"], "after_status": new["status"],
                "before_reason_codes": old["reason_codes"],
                "after_reason_codes": new["reason_codes"],
            })
    proposal_transition = None
    if "proposed_use" in first:
        old, new = first["proposed_use"], second["proposed_use"]
        if any(old[key] != new[key] for key in ("status", "reason_codes", "incomplete")):
            proposal_transition = {
                "before_status": old["status"], "after_status": new["status"],
                "before_reason_codes": old["reason_codes"],
                "after_reason_codes": new["reason_codes"],
                "before_incomplete": old["incomplete"],
                "after_incomplete": new["incomplete"],
            }
    report = {
        "schema_version": 1,
        "language": first["language"],
        "taxonomy_status": first["taxonomy_status"],
        "changed_path": changed[0],
        "before_review_suggested": first["review_suggested"],
        "after_review_suggested": second["review_suggested"],
        "decision_changed": bool(transitions) or proposal_transition is not None,
        "transitions": transitions,
    }
    if proposal_transition is not None:
        report["proposed_use_transition"] = proposal_transition
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one declared contract fact without a model or external action")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    try:
        before = json.loads(args.before.read_text(encoding="utf-8"))
        after = json.loads(args.after.read_text(encoding="utf-8"))
        result = contrast_declared_contracts(before, after)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
