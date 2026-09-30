"""Read-only JSON command line interface for declared contract checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .assess import assess_contracts, assemble_report
from .action import assess_proposed_use
from .policy import (BoundaryCard, DisclosureEvidenceCard, GateCard,
                     PermissionCard, PermissionEvent, ReadabilityCard,
                     RoutingCard, SupportContractCard)
from .words import count_words
from .review import CATEGORIES

CARD_FIELDS = {
    "disclosure": {"source_role", "safety_or_support_anchor",
                   "indirect_or_repeated_support_pattern", "explicit_support_request"},
    "readability": {"declared_age", "measured_words", "max_words", "text"},
    "routing": {"principal", "item", "purpose", "proposed_recipient",
                "recipient_roles", "allowed_by_scope"},
    "permission": {"principal", "item", "purpose", "recipient", "at",
                   "proposed_use", "events", "use_required"},
    "boundary": {"allowed_fields", "protected_fields", "required_fields",
                 "proposed_fields", "prior_correction", "renewed_request"},
    "gate": {"review_owner", "model_output_parsed", "accept_unparsed",
             "external_action_branch", "human_approval_required"},
    "support": {"applicable_fields", "passed_fields"},
}
EVENT_FIELDS = {"sequence", "principal", "item", "purpose", "recipient", "action", "expires_at"}
OPTIONAL_CARD_FIELDS = {"readability": {"text"}}


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _fields(value: dict, expected: set[str], name: str, optional: set[str] = frozenset()) -> None:
    if set(value) - expected or expected - optional - set(value):
        raise ValueError(f"{name} fields differ from the declared schema")


def _set(value: Any, name: str) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string array or null")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicates")
    return frozenset(value)


def _text_fields(value: dict, names: tuple[str, ...]) -> None:
    if any(value[name] is not None and not isinstance(value[name], str) for name in names):
        raise ValueError("Role, item, and purpose fields must be strings or null")


def _integer_or_none(value: Any, name: str, *, nullable: bool = True) -> int | None:
    """Use JSON Schema's value-based integer rule at the JSON boundary."""
    if value is None and nullable:
        return None
    if type(value) is int and value >= 0:
        return value
    if type(value) is float and math.isfinite(value) and value >= 0 and value.is_integer():
        return int(value)
    raise ValueError(f"{name} must be a nonnegative integer" + (" or null" if nullable else ""))


def _routing(value: dict) -> RoutingCard:
    _text_fields(value, ("principal", "item", "purpose", "proposed_recipient"))
    roles = _set(value["recipient_roles"], "recipient_roles")
    scopes = value["allowed_by_scope"]
    if scopes is not None:
        if not isinstance(scopes, list):
            raise ValueError("allowed_by_scope must be an array or null")
        parsed = {}
        for scope in scopes:
            scope = _mapping(scope, "scope")
            _fields(scope, {"principal", "item", "purpose", "recipients"}, "scope")
            key = (scope["principal"], scope["item"], scope["purpose"])
            if not all(isinstance(part, str) for part in key) or key in parsed:
                raise ValueError("Scope key must be unique and contain strings")
            parsed[key] = _set(scope["recipients"], "scope recipients")
            if parsed[key] is None:
                raise ValueError("Scope recipients cannot be null")
        scopes = parsed
    return RoutingCard(value["principal"], value["item"], value["purpose"],
                       value["proposed_recipient"], roles, scopes)


def _permission(value: dict) -> PermissionCard:
    _text_fields(value, ("principal", "item", "purpose", "recipient"))
    events = value["events"]
    if not isinstance(events, list):
        raise ValueError("events must be an array")
    parsed = []
    for event in events:
        event = _mapping(event, "event")
        _fields(event, EVENT_FIELDS, "event", {"expires_at"})
        if any(not isinstance(event[name], str) for name in
               ("principal", "item", "purpose", "recipient", "action")):
            raise ValueError("Invalid permission event value")
        normalized = dict(event)
        normalized["sequence"] = _integer_or_none(event["sequence"], "event sequence", nullable=False)
        if "expires_at" in event:
            normalized["expires_at"] = _integer_or_none(event["expires_at"], "event expiry")
        parsed.append(PermissionEvent(**normalized))
    if (type(value["proposed_use"]) is not bool or
            type(value["use_required"]) is not bool):
        raise ValueError("Invalid permission proposal or time")
    at = _integer_or_none(value["at"], "permission time")
    return PermissionCard(value["principal"], value["item"], value["purpose"],
                          value["recipient"], at, value["proposed_use"], tuple(parsed),
                          value["use_required"])


def _boundary(value: dict) -> BoundaryCard:
    fields = [_set(value[name], name) for name in ("allowed_fields", "protected_fields",
                                                   "required_fields", "proposed_fields")]
    if type(value["prior_correction"]) is not bool or type(value["renewed_request"]) is not bool:
        raise ValueError("Boundary flags must be booleans")
    return BoundaryCard(*fields, value["prior_correction"], value["renewed_request"])


def _gate(value: dict) -> GateCard:
    _text_fields(value, ("review_owner",))
    flags = [value[name] for name in ("model_output_parsed", "accept_unparsed",
                                      "external_action_branch", "human_approval_required")]
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("Gate flags must be booleans")
    return GateCard(value["review_owner"], *flags)


def assess_json(payload: Any) -> dict:
    """Convert one declared, fictional contract object to a serializable report."""
    payload = _mapping(payload, "input")
    if set(payload) - {"language", "proposed_use", *CARD_FIELDS} or "language" not in payload:
        raise ValueError("Input must declare language and supported card fields only")
    language = payload["language"]
    if language not in ("ro", "uk"):
        raise ValueError("language must be ro or uk")
    cards = {}
    for name in CARD_FIELDS:
        if name not in payload:
            continue
        value = _mapping(payload[name], name)
        _fields(value, CARD_FIELDS[name], name, OPTIONAL_CARD_FIELDS.get(name, frozenset()))
        if name == "disclosure":
            role = value["source_role"]
            flags = (value["safety_or_support_anchor"],
                     value["indirect_or_repeated_support_pattern"],
                     value["explicit_support_request"])
            if role is not None and (not isinstance(role, str) or role not in {"minor", "adult", "other"}):
                raise ValueError("Unknown declared source role")
            if any(flag is not None and type(flag) is not bool for flag in flags):
                raise ValueError("Disclosure evidence flags must be booleans or null")
            cards[name] = DisclosureEvidenceCard(**value)
        elif name == "readability":
            age = _integer_or_none(value["declared_age"], "declared_age")
            measured = _integer_or_none(value["measured_words"], "measured_words")
            maximum = _integer_or_none(value["max_words"], "max_words")
            if "text" in value:
                counted = count_words(value["text"])
                if measured is not None and measured != counted:
                    raise ValueError("Declared word count differs from local count")
                measured = counted
            cards[name] = ReadabilityCard(age, measured, maximum)
        elif name == "routing":
            cards[name] = _routing(value)
        elif name == "permission":
            cards[name] = _permission(value)
        elif name == "boundary":
            cards[name] = _boundary(value)
        elif name == "gate":
            cards[name] = _gate(value)
        else:
            cards[name] = SupportContractCard(_set(value["applicable_fields"], "applicable_fields"),
                                              _set(value["passed_fields"], "passed_fields"))
    if not cards and "proposed_use" not in payload:
        raise ValueError("Supply at least one declared contract")
    report = assemble_report(contracts=assess_contracts(**cards)) if cards else None
    findings = report.contracts.findings if report is not None else ()
    result = {
        "schema_version": 2,
        "language": language,
        "taxonomy_status": "approved_ro_v0.2" if language == "ro" else "draft_uk_v0.1",
        "checked_categories": [code for code in CATEGORIES if any(
            finding.category == code for finding in findings)],
        "caller_facts_verified": False,
        "language_verified": False,
        "review_suggested": report.review_suggested if report is not None else False,
        "incomplete": report.contracts.incomplete if report is not None else False,
        "findings": [
            {"category": finding.category, "source_kind": finding.source_kind,
             "decision_code": finding.decision_code,
             "status": finding.status, "reason_codes": list(finding.reason_codes),
             "evidence_basis": finding.evidence_basis}
            for finding in findings
        ],
    }
    if "proposed_use" in payload:
        proposal = _mapping(payload["proposed_use"], "proposed_use")
        _fields(proposal, {"external_action"}, "proposed_use")
        assessment = assess_proposed_use(
            routing=cards.get("routing"), permission=cards.get("permission"),
            gate=cards.get("gate"), external_action=proposal["external_action"])
        result["proposed_use"] = {
            "status": assessment.status,
            "reason_codes": list(assessment.reason_codes),
            "incomplete": assessment.incomplete,
            "action_taken": False,
            "caller_facts_verified": False,
        }
        result["review_suggested"] = True
        result["incomplete"] = result["incomplete"] or assessment.incomplete
    return result


def assess_jsonl(raw: str) -> list[dict]:
    """Validate an entire local stream before emitting any result."""
    lines = raw.splitlines()
    if not lines:
        raise ValueError("JSONL input is empty")
    reports = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"JSONL line {number} is blank")
        try:
            reports.append(assess_json(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"JSONL line {number}: {exc}") from exc
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Check declared fictional contracts; no external action")
    parser.add_argument("input", nargs="?", default="-", help="JSON file, or - for standard input")
    parser.add_argument("--jsonl", action="store_true", help="Assess newline-delimited JSON objects in one validated batch")
    args = parser.parse_args()
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        result = assess_jsonl(raw) if args.jsonl else assess_json(json.loads(raw))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.jsonl:
        for report in result:
            print(json.dumps(report, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
