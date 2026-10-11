"""Exact, text-free decision audit with field-level unknowns.

The audit enumerates declared symbolic completions. Its question order is a
minimax ranking over *decision codes*, not a risk score or a probability.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from .cli import _strict_json_loads
from .policy import SUPPORT_FIELDS, SupportContractCard
from .policy_v1 import (DisclosureCardV1, V1_SUPPORT_FIELDS,
                        check_disclosure_v1, check_support_v1)
from .v1_contract import _finding, assess_v1_json

_D1_FIELDS = ("safety_or_support_anchor", "indirect_or_repeated_support_pattern",
              "explicit_support_request", "direct_safety_statement", "retracted_or_denied")
_S1_FIELDS = tuple(sorted(SUPPORT_FIELDS | V1_SUPPORT_FIELDS))
_S1_STATES = ("not_applicable", "failed", "passed")


def _state(value: object) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    assert isinstance(value, str)
    return value


def _summarize(category: str, current: str | None, paths: list[str],
               completions: list[tuple[tuple[object, ...], str]]) -> dict:
    outcomes = sorted({decision for _, decision in completions})
    witnesses = []
    for decision in outcomes:
        values = next(values for values, code in completions if code == decision)
        witnesses.append({"decision_code": decision,
                          "unknown_facts": [{"path": path, "state_code": _state(value)}
                                            for path, value in zip(paths, values)]})
    questions = []
    if len(outcomes) > 1:
        for index, path in enumerate(paths):
            branches: dict[object, set[str]] = {}
            sizes: dict[object, int] = {}
            for values, decision in completions:
                answer = values[index]
                branches.setdefault(answer, set()).add(decision)
                sizes[answer] = sizes.get(answer, 0) + 1
            # A field that leaves the same decision set for every answer
            # cannot help distinguish the decision at this step.
            if all(set(outcomes) == codes for codes in branches.values()):
                continue
            questions.append({"path": path,
                              "answer_codes": sorted(_state(answer) for answer in branches),
                              "branches": [{"answer_code": _state(answer),
                                            "possible_decision_codes": sorted(codes),
                                            "completion_count": sizes[answer]}
                                           for answer, codes in sorted(
                                               branches.items(), key=lambda item: _state(item[0]))],
                              "worst_case_decision_count": max(map(len, branches.values())),
                              "total_branch_decision_count": sum(map(len, branches.values())),
                              "worst_case_completion_count": max(sizes.values())})
        questions.sort(key=lambda row: (row["worst_case_decision_count"],
                                        row["total_branch_decision_count"],
                                        row["worst_case_completion_count"], row["path"]))
    return {"category": category, "current_decision_code": current,
            "possible_decision_codes": outcomes, "completion_count": len(completions),
            "unknown_paths": paths,
            "decision_stability": "stable" if len(outcomes) == 1 else "sensitive",
            "witnesses": witnesses, "next_questions": questions}


def _disclosure(card: dict, language: str) -> dict:
    current = assess_v1_json({"language": language, "disclosure": card})["checks"][0]["decision_code"]
    paths = []
    if card["source_role"] is None:
        paths.append("/disclosure/source_role")
    paths.extend(f"/disclosure/{field}" for field in _D1_FIELDS if card.get(field) is None)
    roles = ("minor", "adult", "other") if card["source_role"] is None else (card["source_role"],)
    choices = [roles] + [((False, True) if card.get(field) is None else (card[field],))
                         for field in _D1_FIELDS]
    completions = []
    for role, *flags in product(*choices):
        reasons = check_disclosure_v1(DisclosureCardV1(role, *flags)).reason_codes
        values = ((role,) if card["source_role"] is None else ()) + tuple(
            flag for field, flag in zip(_D1_FIELDS, flags) if card.get(field) is None)
        completions.append((values, _finding("D1", reasons)["decision_code"]))
    return _summarize("D1", current, paths, completions)


def _routing(card: dict, language: str) -> dict:
    current = assess_v1_json({"language": language, "routing": card})["checks"][0]["decision_code"]
    missing = [f"/routing/{field}" for field in (
        "principal", "item", "purpose", "proposed_recipient", "recipient_roles",
        "allowed_by_scope") if card[field] is None]
    concern_path = "/routing/recipient_may_be_source_of_concern"
    unknown_concern = card["recipient_may_be_source_of_concern"] == "unknown"
    if missing:
        if unknown_concern:
            missing.append(concern_path)
        return {"category": "R1", "current_decision_code": current,
                "possible_decision_codes": [], "completion_count": None,
                "unknown_paths": missing, "decision_stability": "unbounded",
                "witnesses": [], "next_questions": []}
    if not unknown_concern:
        return _summarize("R1", current, [], [((), current)])
    completions = []
    for concern in ("no", "yes"):
        filled = dict(card)
        filled["recipient_may_be_source_of_concern"] = concern
        decision = assess_v1_json({"language": language, "routing": filled})["checks"][0]["decision_code"]
        completions.append(((concern,), decision))
    return _summarize("R1", current, [concern_path], completions)


def _support(card: object) -> dict:
    if not isinstance(card, dict) or set(card) != {"field_states"}:
        raise ValueError("Support must contain only field_states")
    states = card["field_states"]
    if not isinstance(states, dict) or any(type(key) is not str for key in states):
        raise ValueError("Support field_states must be an object")
    if set(states) - set(_S1_FIELDS) or any(
            state not in (*_S1_STATES, "unknown") or type(state) is not str
            for state in states.values()):
        raise ValueError("Unknown support field or state")
    choices = [(_S1_STATES if states.get(field, "unknown") == "unknown"
                else (states[field],)) for field in _S1_FIELDS]
    paths = [f"/support/field_states/{field}" for field in _S1_FIELDS
             if states.get(field, "unknown") == "unknown"]
    completions = []
    for values in product(*choices):
        applicable = frozenset(field for field, state in zip(_S1_FIELDS, values)
                               if state != "not_applicable")
        passed = frozenset(field for field, state in zip(_S1_FIELDS, values)
                          if state == "passed")
        decision = _finding("S1", check_support_v1(
            SupportContractCard(applicable, passed)).reason_codes)["decision_code"]
        unknown_values = tuple(value for field, value in zip(_S1_FIELDS, values)
                               if states.get(field, "unknown") == "unknown")
        completions.append((unknown_values, decision))
    current = completions[0][1] if not paths else None
    return _summarize("S1", current, paths, completions)


def audit_v2_contract(payload: object) -> dict:
    """Return all reachable decisions and useful questions for symbolic facts."""
    if not isinstance(payload, dict) or (set(payload) -
            {"language", "disclosure", "routing", "support"}) or not (
            set(payload) & {"disclosure", "routing", "support"}):
        raise ValueError("Expected language and at least one v2 card")
    if payload.get("language") not in ("ro", "uk"):
        raise ValueError("Language must be ro or uk")
    language = payload["language"]
    rows = []
    if "disclosure" in payload:
        rows.append(_disclosure(payload["disclosure"], language))
    if "routing" in payload:
        rows.append(_routing(payload["routing"], language))
    if "support" in payload:
        rows.append(_support(payload["support"]))
    return {"schema_version": "2.0", "language": language,
            "method": "exact_symbolic_completion_and_minimax_question_ranking",
            "categories": rows, "caller_facts_unverified": True,
            "external_action_taken": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit corrected symbolic facts and rank unresolved questions")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        report = audit_v2_contract(_strict_json_loads(args.input.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
