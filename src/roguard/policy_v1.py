"""Candidate v1 rules for declared abstract cards; historical rules remain frozen."""

from __future__ import annotations

from dataclasses import dataclass

from .policy import PolicyResult, RoutingCard, SupportContractCard, check_routing, check_support_contract


@dataclass(frozen=True)
class DisclosureCardV1:
    source_role: str | None
    safety_or_support_anchor: bool | None
    indirect_or_repeated_support_pattern: bool | None
    explicit_support_request: bool | None
    direct_safety_statement: bool | None
    retracted_or_denied: bool | None = None


def check_disclosure_v1(card: DisclosureCardV1) -> PolicyResult:
    """A direct safety statement and a request are independent positive paths.

    Retraction is recorded for review but never erases a prior positive path.
    All facts are supplied by the caller; no message text is interpreted here.
    """
    if card.source_role not in {None, "minor", "adult", "other"}:
        raise ValueError("Unknown declared source role")
    flags = (card.safety_or_support_anchor, card.indirect_or_repeated_support_pattern,
             card.explicit_support_request, card.direct_safety_statement,
             card.retracted_or_denied)
    if any(value is not None and type(value) is not bool for value in flags):
        raise ValueError("Disclosure evidence flags must be booleans or null")
    if card.source_role is None:
        return PolicyResult("D1", ("MISSING_SIGNAL_EVIDENCE",))
    if card.source_role != "minor":
        return PolicyResult("D1", ())
    if card.direct_safety_statement:
        return PolicyResult("D1", ("DECLARED_DIRECT_SAFETY_STATEMENT",))
    if card.explicit_support_request:
        return PolicyResult("D1", ("DECLARED_EXPLICIT_SUPPORT",))
    if card.safety_or_support_anchor and card.indirect_or_repeated_support_pattern:
        return PolicyResult("D1", ("DECLARED_INDIRECT_SUPPORT",))
    if any(value is None for value in flags[:4]):
        return PolicyResult("D1", ("MISSING_SIGNAL_EVIDENCE",))
    if not card.safety_or_support_anchor:
        return PolicyResult("D1", ())
    return PolicyResult("D1", ())


@dataclass(frozen=True)
class RoutingCardV1:
    legacy_card: RoutingCard
    recipient_may_be_source_of_concern: str


def check_routing_v1(card: RoutingCardV1) -> PolicyResult:
    """Never call a route aligned while the recipient concern is unresolved."""
    if card.recipient_may_be_source_of_concern not in {"yes", "no", "unknown"}:
        raise ValueError("Recipient concern must be yes, no, or unknown")
    old = check_routing(card.legacy_card)
    if card.recipient_may_be_source_of_concern != "no":
        return PolicyResult("R1", tuple(dict.fromkeys(
            (*old.reason_codes, "SOURCE_OF_CONCERN_UNRESOLVED"))))
    return old


V1_SUPPORT_FIELDS = frozenset({"immediate_danger_route", "no_pressing_for_details"})


def check_support_v1(card: SupportContractCard) -> PolicyResult:
    """The extra fields apply only when the caller declares them applicable."""
    applicable, passed = card.applicable_fields, card.passed_fields
    if applicable is None or passed is None or not applicable:
        return PolicyResult("S1", ("INSUFFICIENT_RESPONSE_CONTRACT",))
    if not isinstance(applicable, frozenset) or not isinstance(passed, frozenset):
        raise ValueError("Support fields must be frozensets or null")
    extra_applicable = applicable & V1_SUPPORT_FIELDS
    extra_passed = passed & V1_SUPPORT_FIELDS
    old = check_support_contract(SupportContractCard(
        applicable - V1_SUPPORT_FIELDS, passed - V1_SUPPORT_FIELDS))
    if not passed <= applicable:
        raise ValueError("Passed support fields must be applicable")
    if not extra_applicable and old.reason_codes == ("MISSING_SUPPORT_CONTRACT",):
        return PolicyResult("S1", ())
    old_reasons = () if old.reason_codes == ("MISSING_SUPPORT_CONTRACT",) else old.reason_codes
    return PolicyResult("S1", old_reasons + tuple(
        f"SUPPORT_{field.upper()}_FAILED" for field in sorted(extra_applicable - extra_passed)))
