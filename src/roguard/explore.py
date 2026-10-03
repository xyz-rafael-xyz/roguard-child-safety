"""Bounded, content-free sensitivity audit of declared contract decisions."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from .cli import _strict_json_loads, assess_json


def _variant(payload: dict, path: tuple[str, ...], replacement: Any) -> dict:
    changed = copy.deepcopy(payload)
    target = changed
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement
    return changed


def _candidates(payload: dict) -> Iterator[tuple[str, str, str | None, dict]]:
    flags = {
        "disclosure": ("safety_or_support_anchor", "indirect_or_repeated_support_pattern",
                       "explicit_support_request"),
        "boundary": ("prior_correction", "renewed_request"),
        "gate": ("model_output_parsed", "accept_unparsed", "external_action_branch",
                 "human_approval_required"),
        "permission": ("proposed_use", "use_required"),
        "proposed_use": ("external_action",),
    }
    for card, names in flags.items():
        if card not in payload:
            continue
        for name in names:
            value = payload[card][name]
            if type(value) is bool:
                yield f"/{card}/{name}", "toggle", None, _variant(payload, (card, name), not value)
            elif value is None:
                for replacement in (False, True):
                    yield f"/{card}/{name}", "supply_false" if not replacement else "supply_true", None, _variant(
                        payload, (card, name), replacement)

    if "disclosure" in payload:
        current = payload["disclosure"]["source_role"]
        for role in ("minor", "adult", "other"):
            if role != current:
                yield "/disclosure/source_role", "alternative_role", None, _variant(
                    payload, ("disclosure", "source_role"), role)

    if "readability" in payload:
        for name in ("declared_age", "measured_words", "max_words"):
            value = payload["readability"][name]
            if type(value) is int:
                for direction, replacement in (("decrease_one", value - 1), ("increase_one", value + 1)):
                    if replacement >= 0:
                        yield f"/readability/{name}", direction, None, _variant(
                            payload, ("readability", name), replacement)

    if "routing" in payload and payload["routing"]["recipient_roles"] is not None:
        current = payload["routing"]["proposed_recipient"]
        for role in sorted(payload["routing"]["recipient_roles"]):
            if role != current:
                yield "/routing/proposed_recipient", "alternative_recipient", None, _variant(
                    payload, ("routing", "proposed_recipient"), role)

    if "permission" in payload:
        card = payload["permission"]
        if type(card["at"]) is int:
            times = {card["at"] - 1, card["at"] + 1}
            for event in card["events"]:
                for key in ("sequence", "expires_at"):
                    point = event.get(key)
                    if type(point) is int:
                        times.update((point - 1, point, point + 1))
            for at in sorted(time for time in times if time >= 0 and time != card["at"]):
                yield "/permission/at", "alternative_time", None, _variant(
                    payload, ("permission", "at"), at)

    if "support" in payload:
        card = payload["support"]
        if card["applicable_fields"] is not None and card["passed_fields"] is not None:
            for field in sorted(card["applicable_fields"]):
                passed = set(card["passed_fields"])
                present = field in passed
                if present:
                    passed.remove(field)
                else:
                    passed.add(field)
                yield "/support/passed_fields", "remove" if present else "add", field, _variant(
                    payload, ("support", "passed_fields"), sorted(passed))


def _changes(before: dict, after: dict) -> list[dict]:
    old = {(finding["category"], finding["source_kind"]): finding
           for finding in before["findings"]}
    new = {(finding["category"], finding["source_kind"]): finding
           for finding in after["findings"]}
    changed = []
    for key, first in old.items():
        second = new[key]
        if (first["status"], first["reason_codes"]) != (second["status"], second["reason_codes"]):
            changed.append({"category": first["category"], "source_kind": first["source_kind"],
                            "before_status": first["status"], "after_status": second["status"],
                            "before_reason_codes": first["reason_codes"],
                            "after_reason_codes": second["reason_codes"]})
    if "proposed_use" in before:
        first, second = before["proposed_use"], after["proposed_use"]
        if (first["status"], first["reason_codes"], first["incomplete"]) != (
                second["status"], second["reason_codes"], second["incomplete"]):
            changed.append({"category": "proposed_use", "source_kind": "composed_use",
                            "before_status": first["status"], "after_status": second["status"],
                            "before_reason_codes": first["reason_codes"],
                            "after_reason_codes": second["reason_codes"]})
    return changed


def explore_declared_contract(payload: dict, *, limit: int = 128) -> dict:
    """Find observed one-fact decision changes; this is no completeness proof."""
    if type(limit) is not int or not 1 <= limit <= 512:
        raise ValueError("limit must be an integer from 1 to 512")
    baseline = assess_json(payload)
    tested = skipped_invalid = 0
    transitions = []
    candidates = iter(_candidates(payload))
    for path, mutation, field, changed in candidates:
        if tested + skipped_invalid >= limit:
            truncated = True
            break
        try:
            report = assess_json(changed)
        except (TypeError, ValueError):
            skipped_invalid += 1
            continue
        tested += 1
        differences = _changes(baseline, report)
        if differences:
            row = {"path": path, "mutation": mutation, "changes": differences}
            if field is not None:
                row["field_code"] = field
            transitions.append(row)
    else:
        truncated = False
    return {"schema_version": 1, "language": baseline["language"],
            "taxonomy_status": baseline["taxonomy_status"],
            "baseline": baseline, "tested": tested, "skipped_invalid": skipped_invalid,
            "truncated": truncated, "decision_changes": transitions,
            "caller_facts_verified": False, "language_verified": False,
            "scope": "bounded_declared_fact_variants"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore one-fact changes to declared contracts, locally")
    parser.add_argument("input", nargs="?", default="-", help="JSON file, or - for standard input")
    parser.add_argument("--limit", type=int, default=128, help="Maximum candidate variants (1–512)")
    args = parser.parse_args()
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        result = explore_declared_contract(_strict_json_loads(raw), limit=args.limit)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
