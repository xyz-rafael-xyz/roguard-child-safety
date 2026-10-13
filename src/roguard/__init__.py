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

def __getattr__(name):
    modules={"DurableGateway":"durable", "GatewayApp":"service", "Monitor":"observability", "CalibratedDecisions":"uncertainty", "FeatureSupport":"uncertainty", "analyze_calibrated_conversation":"uncertainty", "analyze_long_conversation":"long_context", "repair_proposal":"repair", "ChatTarget":"targets"}
    if name in modules:
        from importlib import import_module
        return getattr(import_module("."+modules[name], __name__), name)
    if name in {"FlowLedger", "ToolBinding", "ToolGateway"}:
        from . import flow
        return getattr(flow, name)
    if name == "minimize_disclosure":
        from .disclosure import minimize_disclosure
        return minimize_disclosure
    if name in {"run_scenario", "minimize_failure", "ReferenceTarget", "recorded_factory"}:
        from . import interaction_lab
        return getattr(interaction_lab, name)
    if name in {"evaluate_consistency", "conversation_consistency"}:
        from . import consistency
        return getattr(consistency, name)
    if name in {"summarize_runs", "tradeoff_frontier"}:
        from . import lab_metrics
        return getattr(lab_metrics, name)

    if name in {"analyze_conversation", "analyze_batch", "compare_reports", "ConversationScorer"}:
        from . import conversation
        return getattr(conversation, name)
    if name == "review_response":
        from .response_review import review_response
        return review_response
    raise AttributeError(name)


__all__ = [
    "DurableGateway", "GatewayApp", "Monitor", "CalibratedDecisions", "FeatureSupport", "analyze_calibrated_conversation", "analyze_long_conversation", "repair_proposal", "ChatTarget",
    "FlowLedger", "ToolBinding", "ToolGateway", "minimize_disclosure", "run_scenario", "minimize_failure",
    "ReferenceTarget", "recorded_factory", "evaluate_consistency", "conversation_consistency", "summarize_runs", "tradeoff_frontier",
    "analyze_conversation", "analyze_batch", "compare_reports", "ConversationScorer", "review_response",
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
