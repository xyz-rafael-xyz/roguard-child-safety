"""Bounded one-fact sensitivity audit for caller-declared corrected v1 cards."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from itertools import islice
from pathlib import Path

from .cli import _strict_json_loads
from .policy import SUPPORT_FIELDS
from .v1_contract import assess_v1_json
from .policy_v1 import V1_SUPPORT_FIELDS

MAX_CANDIDATES = 128
TRI_STATES = (None, False, True)
D1_FLAGS = ("safety_or_support_anchor", "indirect_or_repeated_support_pattern",
            "explicit_support_request", "direct_safety_statement", "retracted_or_denied")


def _proposals(payload: dict):
    if "disclosure" in payload:
        card = payload["disclosure"]
        for role in (None, "minor", "adult", "other"):
            if role != card["source_role"]:
                yield ("D1", "/disclosure/source_role", "change_source_role", role)
        for field in D1_FLAGS:
            old = card.get(field)
            for value in TRI_STATES:
                if value != old:
                    yield ("D1", f"/disclosure/{field}", "change_evidence_state", value)
    if "routing" in payload:
        card = payload["routing"]
        for recipient in sorted(card["recipient_roles"] or ()):
            if recipient != card["proposed_recipient"]:
                yield ("R1", "/routing/proposed_recipient", "change_recipient", recipient)
        recipient = card["proposed_recipient"]
        if recipient is not None and card["recipient_roles"] is not None:
            yield ("R1", "/routing/recipient_roles", "toggle_recipient_role", recipient)
        if recipient is not None and card["allowed_by_scope"] is not None:
            for index, scope in enumerate(card["allowed_by_scope"]):
                if (scope["principal"], scope["item"], scope["purpose"]) == (
                        card["principal"], card["item"], card["purpose"]):
                    yield ("R1", f"/routing/allowed_by_scope/{index}/recipients",
                           "toggle_scope_permission", (index, recipient))
        for value in ("yes", "no", "unknown"):
            if value != card["recipient_may_be_source_of_concern"]:
                yield ("R1", "/routing/recipient_may_be_source_of_concern",
                       "change_concern_state", value)
    if "support" in payload:
        card = payload["support"]
        if card["applicable_fields"] is not None and card["passed_fields"] is not None:
            for field in sorted(card["applicable_fields"]):
                yield ("S1", f"/support/passed_fields/{field}", "toggle_passed_field", field)
            for field in sorted(SUPPORT_FIELDS | V1_SUPPORT_FIELDS):
                if field not in card["passed_fields"]:
                    yield ("S1", f"/support/applicable_fields/{field}",
                           "toggle_applicable_field", field)


def _mutate(copy: dict, path: str, mutation: str, value: object) -> None:
    section, field, *rest = path.strip("/").split("/")
    if mutation in {"toggle_passed_field", "toggle_applicable_field",
                    "toggle_recipient_role"}:
        passed = set(copy[section][field])
        if value in passed:
            passed.remove(value)
        else:
            passed.add(value)
        copy[section][field] = sorted(passed)
    elif mutation == "toggle_scope_permission":
        index, recipient = value
        recipients = set(copy["routing"]["allowed_by_scope"][index]["recipients"])
        if recipient in recipients:
            recipients.remove(recipient)
        else:
            recipients.add(recipient)
        copy["routing"]["allowed_by_scope"][index]["recipients"] = sorted(recipients)
    else:
        copy[section][field] = value


def explore_v1_contract(payload: object, limit: int = MAX_CANDIDATES) -> dict:
    """Report decision flips without reflecting fictional card values."""
    if type(limit) is not int or not 1 <= limit <= 512:
        raise ValueError("Exploration limit must be from 1 to 512")
    before = assess_v1_json(payload)
    assert isinstance(payload, dict)
    proposals = list(islice(_proposals(payload), limit + 1))
    changes = []
    skipped = 0
    for category, path, mutation, value in proposals[:limit]:
        altered = deepcopy(payload)
        _mutate(altered, path, mutation, value)
        try:
            after = assess_v1_json(altered)
        except (ValueError, TypeError):
            skipped += 1
            continue
        first = next(check for check in before["checks"] if check["category"] == category)
        second = next(check for check in after["checks"] if check["category"] == category)
        if (first["decision_code"], first["reason_codes"]) != (
                second["decision_code"], second["reason_codes"]):
            if mutation == "change_recipient":
                target = f"declared_recipient_{sorted(payload['routing']['recipient_roles']).index(value) + 1}"
            elif mutation == "toggle_passed_field":
                target = "passed" if value not in payload["support"]["passed_fields"] else "failed"
            elif mutation == "toggle_applicable_field":
                target = ("applicable" if value not in payload["support"]["applicable_fields"]
                          else "not_applicable")
            elif mutation == "toggle_recipient_role":
                target = ("declared" if value not in payload["routing"]["recipient_roles"]
                          else "undeclared")
            elif mutation == "toggle_scope_permission":
                index, recipient = value
                target = ("allowed" if recipient not in
                          payload["routing"]["allowed_by_scope"][index]["recipients"]
                          else "not_allowed")
            elif value is None:
                target = "unknown"
            elif type(value) is bool:
                target = "true" if value else "false"
            else:
                target = value
            changes.append({"category": category, "path": path, "mutation": mutation,
                            "target_state_code": target,
                            "before_decision_code": first["decision_code"],
                            "after_decision_code": second["decision_code"],
                            "before_reason_codes": first["reason_codes"],
                            "after_reason_codes": second["reason_codes"]})
    return {"schema_version": before["schema_version"], "language": before["language"],
            "tested": min(len(proposals), limit), "skipped_invalid": skipped,
            "truncated": len(proposals) > limit, "decision_changes": changes,
            "caller_facts_unverified": True, "external_action_taken": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore corrected v1 symbolic contract decisions")
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args()
    try:
        payload = _strict_json_loads(args.input.read_text(encoding="utf-8"))
        report = explore_v1_contract(payload, args.limit)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
