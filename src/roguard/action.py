"""Compose declared checks for one proposed use; never authorize or take an action."""

from __future__ import annotations

from dataclasses import dataclass

from .assess import INCOMPLETE_REASONS, assess_contracts
from .policy import GateCard, PermissionCard, RoutingCard


@dataclass(frozen=True)
class ProposedUseAssessment:
    status: str  # ready_for_human_decision, incomplete, or review
    reason_codes: tuple[str, ...]
    incomplete: bool


def assess_proposed_use(*, routing: RoutingCard | None = None,
                        permission: PermissionCard | None = None,
                        gate: GateCard | None = None,
                        external_action: bool = False) -> ProposedUseAssessment:
    """Require matching routing and permission scopes before presenting one use to a human.

    A ready result only means that the supplied fictional contracts agree. It does
    not verify caller facts, grant real-world authority, or perform the use.
    """
    if type(external_action) is not bool:
        raise ValueError("external_action must be a boolean")
    if permission is not None and permission.proposed_use is not True:
        raise ValueError("A proposed-use assessment requires proposed_use=true")

    reasons = []
    if routing is None:
        reasons.append("MISSING_ROUTING_CARD")
    if permission is None:
        reasons.append("MISSING_PERMISSION_CARD")
    if external_action and gate is None:
        reasons.append("MISSING_GATE_CARD")

    cards = {"routing": routing, "permission": permission, "gate": gate}
    findings = ()
    if any(card is not None for card in cards.values()):
        findings = assess_contracts(**cards).findings
        for finding in findings:
            if finding.status != "pass":
                reasons.extend(f"{finding.category}:{code}" for code in finding.reason_codes)

    if routing is not None and permission is not None:
        route_scope = (routing.principal, routing.item, routing.purpose,
                       routing.proposed_recipient)
        permission_scope = (permission.principal, permission.item,
                            permission.purpose, permission.recipient)
        if all(isinstance(value, str) and value.strip() for value in route_scope + permission_scope):
            if route_scope != permission_scope:
                reasons.append("SCOPE_MISMATCH")
    if gate is not None and gate.external_action_branch != external_action:
        reasons.append("GATE_BRANCH_MISMATCH")

    incomplete = any(code.startswith("MISSING_") for code in reasons) or any(
        code in INCOMPLETE_REASONS for finding in findings for code in finding.reason_codes)
    if "SCOPE_MISMATCH" in reasons or "GATE_BRANCH_MISMATCH" in reasons or any(
            finding.status == "review" for finding in findings):
        status = "review"
    elif incomplete:
        status = "incomplete"
    else:
        status = "ready_for_human_decision"
    return ProposedUseAssessment(status, tuple(reasons), incomplete)
