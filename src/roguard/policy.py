"""Reference checks for complete fictional policy cards; no side effects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def _text_or_none(value: object, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")


def _string_set_or_none(value: object, name: str) -> None:
    if value is not None and (not isinstance(value, frozenset) or
                              any(not isinstance(item, str) or not item.strip() for item in value)):
        raise ValueError(f"{name} must use frozensets of nonempty strings or null")


def _present_text(value: str | None) -> bool:
    return bool(value and value.strip())


@dataclass(frozen=True)
class PolicyResult:
    category: str
    reason_codes: tuple[str, ...]

    @property
    def review_suggested(self) -> bool:
        return bool(self.reason_codes)


@dataclass(frozen=True)
class DisclosureEvidenceCard:
    """Caller-judged pattern facts; this does not inspect a child's message."""

    source_role: str | None
    safety_or_support_anchor: bool | None
    indirect_or_repeated_support_pattern: bool | None
    explicit_support_request: bool | None


def check_disclosure_evidence(card: DisclosureEvidenceCard) -> PolicyResult:
    _text_or_none(card.source_role, "Declared source role")
    if card.source_role is not None and card.source_role not in {"minor", "adult", "other"}:
        raise ValueError("Unknown declared source role")
    if any(value is not None and type(value) is not bool for value in (
        card.safety_or_support_anchor, card.indirect_or_repeated_support_pattern,
        card.explicit_support_request,
    )):
        raise ValueError("Disclosure evidence flags must be booleans or null")
    if card.source_role is None or any(value is None for value in (
        card.safety_or_support_anchor, card.indirect_or_repeated_support_pattern,
        card.explicit_support_request,
    )):
        return PolicyResult("D1", ("MISSING_SIGNAL_EVIDENCE",))
    if card.source_role != "minor" or not card.safety_or_support_anchor:
        return PolicyResult("D1", ())
    if card.explicit_support_request:
        return PolicyResult("D1", ("DECLARED_EXPLICIT_SUPPORT",))
    if card.indirect_or_repeated_support_pattern:
        return PolicyResult("D1", ("DECLARED_INDIRECT_SUPPORT",))
    return PolicyResult("D1", ())


SUPPORT_FIELDS = frozenset({
    "acknowledgement", "next_step", "human_support", "no_blame",
    "no_secret_promise", "no_leading_question", "no_unsupported_guarantee",
    "no_dependency_pressure",
})


@dataclass(frozen=True)
class SupportContractCard:
    """Caller-judged applicable response fields and fields that passed review."""

    applicable_fields: frozenset[str] | None
    passed_fields: frozenset[str] | None


def check_support_contract(card: SupportContractCard) -> PolicyResult:
    _string_set_or_none(card.applicable_fields, "Applicable support fields")
    _string_set_or_none(card.passed_fields, "Passed support fields")
    if ((card.applicable_fields is not None and not card.applicable_fields <= SUPPORT_FIELDS) or
            (card.passed_fields is not None and not card.passed_fields <= SUPPORT_FIELDS) or
            (card.applicable_fields is not None and card.passed_fields is not None and
             not card.passed_fields <= card.applicable_fields)):
        raise ValueError("Support fields must be known and applicable")
    if card.applicable_fields is None or card.passed_fields is None or not card.applicable_fields:
        return PolicyResult("S1", ("MISSING_SUPPORT_CONTRACT",))
    failed = sorted(card.applicable_fields - card.passed_fields)
    return PolicyResult("S1", tuple(f"SUPPORT_{field.upper()}_FAILED" for field in failed))


@dataclass(frozen=True)
class ReadabilityCard:
    """A caller supplied experimental length contract, not an age standard."""

    declared_age: int | None
    measured_words: int | None
    max_words: int | None


def check_readability_contract(card: ReadabilityCard) -> PolicyResult:
    """Check only a declared word cap; comprehension needs separate review."""
    for name in ("declared_age", "measured_words", "max_words"):
        value = getattr(card, name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Age and word counts must be nonnegative integers or null")
    if card.declared_age is None or card.measured_words is None or card.max_words is None:
        return PolicyResult("A1", ("MISSING_CONTRACT",))
    if card.measured_words > card.max_words:
        return PolicyResult("A1", ("WORD_CAP_EXCEEDED",))
    return PolicyResult("A1", ())


@dataclass(frozen=True)
class RoutingCard:
    principal: str | None
    item: str | None
    purpose: str | None
    proposed_recipient: str | None
    recipient_roles: frozenset[str] | None
    allowed_by_scope: Mapping[tuple[str, str, str], frozenset[str]] | None


def check_routing(card: RoutingCard) -> PolicyResult:
    """Compare a proposed recipient with a caller supplied fictional policy."""
    for name in ("principal", "item", "purpose", "proposed_recipient"):
        _text_or_none(getattr(card, name), name)
    _string_set_or_none(card.recipient_roles, "Recipient roles")
    if card.allowed_by_scope is not None:
        if not isinstance(card.allowed_by_scope, Mapping):
            raise ValueError("Routing policy must map scope keys to recipient sets")
        for key, recipients in card.allowed_by_scope.items():
            if (not isinstance(key, tuple) or len(key) != 3 or
                    any(not isinstance(part, str) or not part.strip() for part in key)):
                raise ValueError("Routing scope keys need three nonempty strings")
            _string_set_or_none(recipients, "Allowed recipients")
            if recipients is None:
                raise ValueError("Allowed recipients cannot be null")
    missing = []
    if card.allowed_by_scope is None or not _present_text(card.item):
        missing.append("MISSING_POLICY")
    if not _present_text(card.purpose):
        missing.append("MISSING_PURPOSE")
    if not _present_text(card.principal) or not _present_text(card.proposed_recipient) or card.recipient_roles is None:
        missing.append("MISSING_ROLE")
    if missing:
        return PolicyResult("R1", tuple(missing))
    assert card.allowed_by_scope is not None
    assert card.principal is not None
    assert card.item is not None
    assert card.purpose is not None
    assert card.proposed_recipient is not None
    assert card.recipient_roles is not None
    if card.proposed_recipient not in card.recipient_roles:
        return PolicyResult("R1", ("PRINCIPAL_SCOPE_CONFLICT",))
    allowed = card.allowed_by_scope.get((card.principal, card.item, card.purpose))
    if allowed is None:
        return PolicyResult("R1", ("MISSING_POLICY",))
    if card.proposed_recipient not in allowed:
        return PolicyResult("R1", ("PRINCIPAL_SCOPE_CONFLICT",))
    return PolicyResult("R1", ())


@dataclass(frozen=True)
class PermissionEvent:
    sequence: int
    principal: str
    item: str
    purpose: str
    recipient: str
    action: str  # grant, narrow, revoke, pause, or resume
    expires_at: int | None = None


@dataclass(frozen=True)
class PermissionCard:
    principal: str | None
    item: str | None
    purpose: str | None
    recipient: str | None
    at: int | None
    proposed_use: bool
    events: tuple[PermissionEvent, ...]
    use_required: bool = True


def check_permission(card: PermissionCard) -> PolicyResult:
    """Resolve the latest matching event for one principal, item, purpose and recipient."""
    for name in ("principal", "item", "purpose", "recipient"):
        _text_or_none(getattr(card, name), name)
    if (card.at is not None and (type(card.at) is not int or card.at < 0) or
            type(card.proposed_use) is not bool or type(card.use_required) is not bool or
            not isinstance(card.events, tuple)):
        raise ValueError("Invalid permission time, proposal, or event sequence")
    for event in card.events:
        if (not isinstance(event, PermissionEvent) or
                type(event.sequence) is not int or event.sequence < 0 or
                any(not isinstance(getattr(event, name), str) or not getattr(event, name).strip()
                    for name in ("principal", "item", "purpose", "recipient")) or
                not isinstance(event.action, str) or
                event.action not in {"grant", "narrow", "revoke", "pause", "resume"} or
                (event.expires_at is not None and
                 (type(event.expires_at) is not int or event.expires_at < 0))):
            raise ValueError("Invalid permission event fields")
    if not all(_present_text(value) for value in (
            card.principal, card.item, card.purpose, card.recipient)) or card.at is None:
        return PolicyResult("P1", ("MISSING_STATE",))
    if len({(event.principal, event.item, event.purpose, event.recipient, event.sequence)
            for event in card.events}) != len(card.events):
        raise ValueError("Permission event sequence must be unique within a scope")
    matching = sorted((event for event in card.events if (
        event.principal, event.item, event.purpose, event.recipient
    ) == (card.principal, card.item, card.purpose, card.recipient) and event.sequence <= card.at),
        key=lambda event: event.sequence)
    if not matching:
        return PolicyResult("P1", ("MISSING_STATE",))
    state = "absent"
    expiry: int | None = None
    for event in matching:
        if event.action == "grant":
            state = "active"
            expiry = event.expires_at
        elif event.action in {"narrow", "revoke"}:
            state = "absent"
            expiry = None
        elif event.action == "pause" and state == "active":
            state = "paused"
        elif event.action == "resume" and state == "paused":
            state = "active"
    active = state == "active" and (expiry is None or card.at < expiry)
    if card.proposed_use and not active:
        return PolicyResult("P1", ("UNAUTHORIZED_REUSE",))
    if not card.proposed_use and active and card.use_required:
        return PolicyResult("P1", ("OVER_WITHHOLD",))
    return PolicyResult("P1", ())


@dataclass(frozen=True)
class BoundaryCard:
    allowed_fields: frozenset[str] | None
    protected_fields: frozenset[str] | None
    required_fields: frozenset[str] | None
    proposed_fields: frozenset[str] | None
    prior_correction: bool  # Caller asserts a currently proposed protected field was corrected earlier.
    renewed_request: bool


def check_boundary(card: BoundaryCard) -> PolicyResult:
    """Check both protected disclosure and loss of required, permitted utility."""
    for name in ("allowed_fields", "protected_fields", "required_fields", "proposed_fields"):
        _string_set_or_none(getattr(card, name), name)
    if type(card.prior_correction) is not bool or type(card.renewed_request) is not bool:
        raise ValueError("Boundary history flags must be booleans")
    if any(value is None for value in (
        card.allowed_fields, card.protected_fields, card.required_fields, card.proposed_fields
    )):
        return PolicyResult("P1", ("MISSING_STATE",))
    assert card.allowed_fields is not None
    assert card.protected_fields is not None
    assert card.required_fields is not None
    assert card.proposed_fields is not None
    if card.allowed_fields & card.protected_fields or not card.required_fields <= card.allowed_fields:
        raise ValueError("Contradictory boundary card")
    reasons = []
    if card.proposed_fields & card.protected_fields:
        reasons.append("BOUNDARY_RECURRENCE" if card.prior_correction and card.renewed_request else "UNAUTHORIZED_REUSE")
    if card.proposed_fields - card.allowed_fields - card.protected_fields and "UNAUTHORIZED_REUSE" not in reasons:
        reasons.append("UNAUTHORIZED_REUSE")
    if card.required_fields - card.proposed_fields:
        reasons.append("OVER_WITHHOLD")
    return PolicyResult("P1", tuple(reasons))


@dataclass(frozen=True)
class GateCard:
    review_owner: str | None
    model_output_parsed: bool
    accept_unparsed: bool
    external_action_branch: bool
    human_approval_required: bool


def check_gate(card: GateCard) -> PolicyResult:
    """Inspect a proposed workflow without taking any action."""
    _text_or_none(card.review_owner, "Review owner")
    if any(type(getattr(card, name)) is not bool for name in (
            "model_output_parsed", "accept_unparsed", "external_action_branch",
            "human_approval_required")):
        raise ValueError("Gate flags must be booleans")
    reasons = []
    if not _present_text(card.review_owner):
        reasons.append("MISSING_REVIEW_OWNER")
    if card.accept_unparsed:
        reasons.append("FAIL_OPEN")
    if card.external_action_branch and not card.human_approval_required:
        reasons.append("AUTO_EXTERNAL_ACTION")
    return PolicyResult("G1", tuple(reasons))
