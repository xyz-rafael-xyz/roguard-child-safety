"""Combine declared-contract checks without taking or authorizing an action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .policy import (BoundaryCard, DisclosureEvidenceCard, GateCard, PermissionCard,
                     PolicyResult, ReadabilityCard, RoutingCard, SupportContractCard,
                     check_boundary, check_disclosure_evidence, check_gate,
                     check_permission, check_readability_contract, check_routing,
                     check_support_contract)
from .review import ACTIVE_LANGUAGES, APPLIES_TO, CATEGORIES, KINDS, LANGUAGES
from .screen import ScoreBackend, ScreenResult, screen

INCOMPLETE_REASONS = frozenset({
    "MISSING_CONTRACT", "MISSING_POLICY", "MISSING_PURPOSE", "MISSING_ROLE",
    "MISSING_STATE", "MISSING_REVIEW_OWNER", "OUTPUT_BLOCKED",
    "MISSING_SIGNAL_EVIDENCE", "MISSING_SUPPORT_CONTRACT",
})
DECISION_CODES = {
    "D1": {"pass": "NO_D1_SIGNAL", "incomplete": "INSUFFICIENT_CONTEXT", "review": "REVIEW_SUPPORT_SIGNAL"},
    "R1": {"pass": "NO_ROUTE_SIGNAL", "incomplete": "REVIEW_ROUTE", "review": "REVIEW_ROUTE"},
    "A1": {"pass": "NO_A1_SIGNAL", "incomplete": "NEEDS_LANGUAGE_CALIBRATION", "review": "REVIEW_ACCESSIBILITY"},
    "P1": {"pass": "NO_PERSISTENCE_SIGNAL", "incomplete": "REVIEW_PERSISTENCE", "review": "REVIEW_PERSISTENCE"},
    "G1": {"pass": "NO_GATE_SIGNAL", "incomplete": "REVIEW_WORKFLOW", "review": "REVIEW_WORKFLOW"},
    "S1": {"pass": "NO_RESPONSE_SIGNAL", "incomplete": "INSUFFICIENT_RESPONSE_CONTRACT", "review": "REVIEW_RESPONSE"},
}
EVIDENCE_BASIS = {
    "message": "caller_judgment",
    "response": "caller_judgment",
    "routing_card": "declared_policy",
    "permission_card": "declared_permission_state",
    "boundary_card": "declared_boundary",
    "gate_card": "declared_workflow",
}


@dataclass(frozen=True)
class ContractFinding:
    category: str
    source_kind: str
    status: str  # pass, incomplete, or review
    reason_codes: tuple[str, ...]
    evidence_basis: str

    @property
    def decision_code(self) -> str:
        return DECISION_CODES[self.category][self.status]


@dataclass(frozen=True)
class ContractAssessment:
    findings: tuple[ContractFinding, ...]

    @property
    def review_suggested(self) -> bool:
        return any(finding.status != "pass" for finding in self.findings)

    @property
    def incomplete(self) -> bool:
        return any(finding.status == "incomplete" for finding in self.findings)


@dataclass(frozen=True)
class GuardReport:
    """Keep model observations separate from declared-contract decisions."""

    contracts: ContractAssessment | None
    advisory_labels: tuple[str, ...]
    unverified_policy_labels: tuple[str, ...]

    @property
    def policy_labels_without_contract(self) -> tuple[str, ...]:
        covered = {finding.category for finding in self.contracts.findings} if self.contracts else set()
        return tuple(code for code in self.unverified_policy_labels if code not in covered)

    @property
    def policy_disagreements(self) -> tuple[str, ...]:
        """Model policy labels opposed by complete passing declared contracts."""
        if not self.contracts:
            return ()
        return tuple(code for code in self.unverified_policy_labels if
                     any(finding.category == code for finding in self.contracts.findings) and
                     all(finding.status == "pass" for finding in self.contracts.findings
                         if finding.category == code))

    @property
    def review_suggested(self) -> bool:
        return (bool(self.advisory_labels) or bool(self.policy_labels_without_contract) or
                bool(self.contracts and self.contracts.review_suggested))


def _finding(source_kind: str, result: PolicyResult) -> ContractFinding:
    if not result.reason_codes:
        status = "pass"
    elif all(reason in INCOMPLETE_REASONS for reason in result.reason_codes):
        status = "incomplete"
    else:
        status = "review"
    basis = "declared_word_cap" if result.category == "A1" else EVIDENCE_BASIS[source_kind]
    return ContractFinding(result.category, source_kind, status, result.reason_codes, basis)


def _gate_with_runtime_state(card: GateCard) -> PolicyResult:
    """Keep policy validity separate from a particular blocked model output."""
    result = check_gate(card)
    if not card.model_output_parsed and not card.accept_unparsed:
        return PolicyResult("G1", result.reason_codes + ("OUTPUT_BLOCKED",))
    return result


def assess_contracts(*, disclosure: DisclosureEvidenceCard | None = None,
                     readability: ReadabilityCard | None = None,
                     routing: RoutingCard | None = None,
                     permission: PermissionCard | None = None,
                     boundary: BoundaryCard | None = None,
                     gate: GateCard | None = None,
                     support: SupportContractCard | None = None) -> ContractAssessment:
    """Return independent, typed findings for the supplied declared contracts."""
    checks = (
        ("message", disclosure, check_disclosure_evidence),
        ("response", readability, check_readability_contract),
        ("routing_card", routing, check_routing),
        ("permission_card", permission, check_permission),
        ("boundary_card", boundary, check_boundary),
        ("gate_card", gate, _gate_with_runtime_state),
        ("response", support, check_support_contract),
    )
    findings = tuple(_finding(kind, checker(card)) for kind, card, checker in checks if card is not None)
    if not findings:
        raise ValueError("Supply at least one structured contract")
    return ContractAssessment(findings)


def assemble_report(*, contracts: ContractAssessment | None = None,
                    model_signal: ScreenResult | None = None) -> GuardReport:
    """Expose D1/A1/S1 as advisory; never substitute model R1/P1/G1 for policy."""
    if contracts is None and model_signal is None:
        raise ValueError("Supply contract findings or a model signal")
    if model_signal is not None:
        if (model_signal.language not in ACTIVE_LANGUAGES or
                model_signal.source_kind not in KINDS or
                len(model_signal.labels) != len(set(model_signal.labels)) or
                any(code not in CATEGORIES or model_signal.source_kind not in APPLIES_TO[code]
                    for code in model_signal.labels)):
            raise ValueError("Model signal has invalid language, category, or source kind")
    advisory = tuple(code for code in ("D1", "A1", "S1")
                     if model_signal is not None and code in model_signal.labels)
    unverified = tuple(code for code in ("R1", "P1", "G1")
                       if model_signal is not None and code in model_signal.labels)
    return GuardReport(contracts, advisory, unverified)


def assess_case(*, language: str, text: str | None = None,
                source_kind: str | None = None, backend: ScoreBackend | None = None,
                thresholds: Mapping[str, float] | None = None,
                disclosure: DisclosureEvidenceCard | None = None,
                readability: ReadabilityCard | None = None,
                routing: RoutingCard | None = None,
                permission: PermissionCard | None = None,
                boundary: BoundaryCard | None = None,
                gate: GateCard | None = None,
                support: SupportContractCard | None = None) -> GuardReport:
    """Assess supplied contracts first, then an optional caller-provided scorer."""
    if language not in LANGUAGES:
        raise ValueError("Language must be ro or uk")
    cards = dict(disclosure=disclosure, readability=readability, routing=routing,
                 permission=permission, boundary=boundary, gate=gate, support=support)
    has_cards = any(card is not None for card in cards.values())
    if text is None:
        if any(value is not None for value in (source_kind, backend, thresholds)):
            raise ValueError("Text, source kind, backend, and thresholds must be supplied together")
    elif source_kind is None or backend is None or thresholds is None:
        raise ValueError("Text, source kind, backend, and thresholds must be supplied together")
    if not has_cards and text is None:
        raise ValueError("Supply declared contracts or model text")
    contracts = assess_contracts(**cards) if has_cards else None
    model_signal = (screen(text, language=language, source_kind=source_kind,
                           backend=backend, thresholds=thresholds)
                    if text is not None else None)
    return assemble_report(contracts=contracts, model_signal=model_signal)
