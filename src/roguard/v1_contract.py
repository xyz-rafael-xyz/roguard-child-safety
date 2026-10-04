"""Read-only candidate v1 contract checks on caller-declared symbolic cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import _fields, _mapping, _routing, _set, _strict_json_loads
from .policy import SupportContractCard
from .policy_v1 import (DisclosureCardV1, RoutingCardV1, check_disclosure_v1,
                        check_routing_v1, check_support_v1)

DISCLOSURE = {"source_role", "safety_or_support_anchor",
              "indirect_or_repeated_support_pattern", "explicit_support_request",
              "direct_safety_statement", "retracted_or_denied"}
ROUTING = {"principal", "item", "purpose", "proposed_recipient", "recipient_roles",
           "allowed_by_scope", "recipient_may_be_source_of_concern"}
SUPPORT = {"applicable_fields", "passed_fields"}


def _finding(category: str, reasons: tuple[str, ...]) -> dict:
    if category == "D1":
        decision = ("INSUFFICIENT_CONTEXT" if "MISSING_SIGNAL_EVIDENCE" in reasons else
                    "REVIEW_SUPPORT_SIGNAL" if reasons else "NO_D1_SIGNAL")
    elif category == "R1":
        decision = "REVIEW_ROUTE" if reasons else "NO_ROUTE_SIGNAL"
    else:
        decision = ("INSUFFICIENT_RESPONSE_CONTRACT" if
                    "INSUFFICIENT_RESPONSE_CONTRACT" in reasons else
                    "REVIEW_RESPONSE" if reasons else "NO_RESPONSE_SIGNAL")
    return {"category": category, "decision_code": decision,
            "reason_codes": list(reasons)}


def assess_v1_json(payload: object) -> dict:
    payload = _mapping(payload, "input")
    if set(payload) - {"language", "disclosure", "routing", "support"} or \
            "language" not in payload or not set(payload) & {"disclosure", "routing", "support"}:
        raise ValueError("Expected language and at least one v1 card")
    if payload["language"] not in ("ro", "uk"):
        raise ValueError("Language must be ro or uk")
    checks = []
    if "disclosure" in payload:
        card = _mapping(payload["disclosure"], "disclosure")
        _fields(card, DISCLOSURE, "disclosure", {"retracted_or_denied"})
        result = check_disclosure_v1(DisclosureCardV1(**card))
        checks.append(_finding(result.category, result.reason_codes))
    if "routing" in payload:
        card = _mapping(payload["routing"], "routing")
        _fields(card, ROUTING, "routing")
        concern = card["recipient_may_be_source_of_concern"]
        result = check_routing_v1(RoutingCardV1(
            _routing({key: value for key, value in card.items() if key != "recipient_may_be_source_of_concern"}),
            concern))
        checks.append(_finding(result.category, result.reason_codes))
    if "support" in payload:
        card = _mapping(payload["support"], "support")
        _fields(card, SUPPORT, "support")
        result = check_support_v1(SupportContractCard(
            _set(card["applicable_fields"], "applicable support fields"),
            _set(card["passed_fields"], "passed support fields")))
        checks.append(_finding(result.category, result.reason_codes))
    return {"schema_version": "1.0-candidate", "language": payload["language"],
            "checks": checks, "caller_facts_unverified": True, "external_action_taken": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check declared abstract cards with candidate v1 rules")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        payload = _strict_json_loads(args.input.read_text(encoding="utf-8"))
        report = assess_v1_json(payload)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
