"""Exact finite completion audit for unknown facts in corrected v1 cards."""

from __future__ import annotations

import argparse
import json
from itertools import product
from math import prod
from pathlib import Path

from .cli import _strict_json_loads
from .policy import SUPPORT_FIELDS, SupportContractCard
from .policy_v1 import (DisclosureCardV1, V1_SUPPORT_FIELDS,
                        check_disclosure_v1, check_support_v1)
from .v1_contract import _finding, assess_v1_json

_D1_FIELDS = ("safety_or_support_anchor", "indirect_or_repeated_support_pattern",
              "explicit_support_request", "direct_safety_statement", "retracted_or_denied")
_S1_FIELDS = tuple(sorted(SUPPORT_FIELDS | V1_SUPPORT_FIELDS))


def _row(category: str, current: str, outcomes: set[str], count: int,
         unknown: list[str], *, bounded: bool = True) -> dict:
    return {"category": category, "current_decision_code": current,
            "possible_decision_codes": sorted(outcomes),
            "completion_count": count if bounded else None,
            "unknown_paths": unknown,
            "decision_stability": ("unbounded" if not bounded else
                                   "stable" if len(outcomes) == 1 else "sensitive")}


def _d1(card: dict, current: str) -> dict:
    roles = ("minor", "adult", "other") if card["source_role"] is None else (card["source_role"],)
    choices = [(False, True) if card.get(field) is None else (card[field],)
               for field in _D1_FIELDS]
    outcomes = set()
    for role in roles:
        for values in product(*choices):
            reasons = check_disclosure_v1(DisclosureCardV1(role, *values)).reason_codes
            outcomes.add(_finding("D1", reasons)["decision_code"])
    unknown = (["/disclosure/source_role"] if card["source_role"] is None else [])
    unknown += [f"/disclosure/{field}" for field in _D1_FIELDS if card.get(field) is None]
    return _row("D1", current, outcomes, len(roles) *
                prod(len(options) for options in choices), unknown)


def _r1(card: dict, current: str) -> dict:
    unknown = [f"/routing/{field}" for field in (
        "principal", "item", "purpose", "proposed_recipient", "recipient_roles",
        "allowed_by_scope") if card[field] is None]
    if unknown:
        if card["recipient_may_be_source_of_concern"] == "unknown":
            unknown.append("/routing/recipient_may_be_source_of_concern")
        return _row("R1", current, set(), 0, unknown, bounded=False)
    if card["recipient_may_be_source_of_concern"] != "unknown":
        return _row("R1", current, {current}, 1, [])
    # With all policy fields fixed, only the explicit concern state is open.
    from .v1_contract import assess_v1_json
    outcomes = set()
    for concern in ("yes", "no"):
        completed = {"language": "ro", "routing": dict(card)}
        completed["routing"]["recipient_may_be_source_of_concern"] = concern
        outcomes.add(assess_v1_json(completed)["checks"][0]["decision_code"])
    return _row("R1", current, outcomes, 2,
                ["/routing/recipient_may_be_source_of_concern"])


def _s1(card: dict, current: str) -> dict:
    applicable = card["applicable_fields"]
    passed = card["passed_fields"]
    unknown = [f"/support/{field}" for field in ("applicable_fields", "passed_fields")
               if card[field] is None]
    if not unknown:
        return _row("S1", current, {current}, 1, [])
    app_fixed = frozenset(applicable) if applicable is not None else None
    pass_fixed = frozenset(passed) if passed is not None else None
    options = []
    for field in _S1_FIELDS:
        if app_fixed is not None:
            if field not in app_fixed:
                options.append((0,))
            elif pass_fixed is None:
                options.append((1, 2))
            else:
                options.append((2,) if field in pass_fixed else (1,))
        elif pass_fixed is not None and field in pass_fixed:
            options.append((2,))
        else:
            options.append((0, 1, 2))
    outcomes = set()
    count = 0
    for states in product(*options):
        app = frozenset(field for field, state in zip(_S1_FIELDS, states) if state)
        pas = frozenset(field for field, state in zip(_S1_FIELDS, states) if state == 2)
        reasons = check_support_v1(SupportContractCard(app, pas)).reason_codes
        outcomes.add(_finding("S1", reasons)["decision_code"])
        count += 1
    return _row("S1", current, outcomes, count, unknown)


def envelope_v1_contract(payload: object) -> dict:
    """Enumerate finite completions; do not infer any fact or expose card values.

    Stability means the decision code is invariant over these completions only.
    Missing routing strings/policies have no finite declared domain, so no
    stability certificate is returned for that category.
    """
    baseline = assess_v1_json(payload)
    assert isinstance(payload, dict)
    current = {row["category"]: row["decision_code"] for row in baseline["checks"]}
    rows = []
    if "disclosure" in payload:
        rows.append(_d1(payload["disclosure"], current["D1"]))
    if "routing" in payload:
        rows.append(_r1(payload["routing"], current["R1"]))
    if "support" in payload:
        rows.append(_s1(payload["support"], current["S1"]))
    return {"schema_version": "1.5", "language": baseline["language"],
            "method": "exact_finite_completion_of_declared_unknowns",
            "categories": rows, "caller_facts_unverified": True,
            "external_action_taken": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit unknown facts in corrected v1 symbolic cards")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        report = envelope_v1_contract(_strict_json_loads(args.input.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
