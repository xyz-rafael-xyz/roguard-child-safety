"""Research-only signal classification. No external actions are performed."""

from .screen import CATEGORIES, ScreenResult, screen
from .calibrate import calibrate
from .escalate import HumanReviewSignal, escalate
from .model import HFClassifier, MLXClassifier
from .token_margin import MLXTokenMarginClassifier
from .mmbert_adapter import MMBertResearchClassifier
from .factor_adapter import D1FactorResearchClassifier
from .joint_adapter import JointD1ResearchClassifier
from .balanced_adapter import BalancedJointD1ResearchClassifier
from .nli_adapter import NLIResearchClassifier
from .words import count_words
from .assess import (ContractAssessment, ContractFinding, GuardReport,
                     assemble_report, assess_case, assess_contracts)
from .action import ProposedUseAssessment, assess_proposed_use
from .agreement import compare_evidence
from .review_card import collect_disclosure_card, collect_support_card
from .agreement_batch import compare_evidence_batches
from .policy import (BoundaryCard, DisclosureEvidenceCard, GateCard, PermissionCard,
                     PermissionEvent, PolicyResult, ReadabilityCard, RoutingCard,
                     SupportContractCard, check_boundary, check_disclosure_evidence,
                     check_gate, check_permission, check_readability_contract,
                     check_routing, check_support_contract)

__all__ = [
    "CATEGORIES", "ScreenResult", "HumanReviewSignal", "screen", "calibrate", "escalate",
    "HFClassifier", "MLXClassifier", "MLXTokenMarginClassifier", "MMBertResearchClassifier", "D1FactorResearchClassifier", "JointD1ResearchClassifier", "BalancedJointD1ResearchClassifier", "NLIResearchClassifier", "count_words",
    "ContractAssessment", "ContractFinding", "GuardReport", "assess_contracts", "assemble_report", "assess_case",
    "ProposedUseAssessment", "assess_proposed_use",
    "compare_evidence",
    "collect_disclosure_card", "collect_support_card",
    "compare_evidence_batches",
    "PolicyResult", "DisclosureEvidenceCard", "ReadabilityCard", "RoutingCard", "PermissionCard", "PermissionEvent", "BoundaryCard", "GateCard", "SupportContractCard",
    "check_disclosure_evidence", "check_readability_contract", "check_routing", "check_permission", "check_boundary", "check_gate", "check_support_contract",
]
