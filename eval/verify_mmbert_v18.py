"""Recompute the frozen V18 diagnostic test from saved factor and model scores."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v18-test-0034.json")
MODEL = Path("models/ro-mmbert-v18-factor-abstract")
BASELINE = Path("models/ro-mmbert-v16-abstract")


def verify(root: Path = ROOT) -> dict:
    logic = runpy.run_path(str(root / "eval/run_mmbert_v18.py"))
    selection = runpy.run_path(str(root / "eval/verify_committed_v18_selection.py"))["verify"](root)
    facts = runpy.run_path(str(root / "training/v18_facts.py"))[
        "load_d1_rows_with_facts"](root, "batch-0034")
    rows = load_approved(root, ["batch-0034"])
    run = json.loads((root / RUN).read_text(encoding="utf-8"))
    scores = run.get("scores")
    if (run.get("status") != "frozen_v18_factor_test" or
            run.get("batch") != "batch-0034" or len(rows) != 144 or
            run.get("batch_sha256") != sha256(root / "data/synthetic/batch-0034.jsonl") or
            run.get("attestation_sha256") != sha256(root / "data/synthetic/batch-0034.review.json") or
            run.get("protocol_sha256") != sha256(root / "docs/V18_PROTOCOL.md") or
            run.get("prefit_amendment_sha256") != sha256(root / "docs/V18_PREFIT_AMENDMENT.md") or
            run.get("runner_sha256") != sha256(root / "eval/run_mmbert_v18.py") or
            run.get("selection_record_sha256") != sha256(root / "eval/runs/ro-mmbert-v18-dev-selection.json") or
            run.get("v18_manifest_sha256") != sha256(root / MODEL / "research.json") or
            run.get("v18_weight_sha256") != sha256(root / MODEL / "adapter_model.safetensors") or
            run.get("v16_manifest_sha256") != sha256(root / BASELINE / "research.json") or
            run.get("v16_weight_sha256") != sha256(root / BASELINE / "adapter_model.safetensors") or
            run.get("selected_development") != selection or
            not isinstance(scores, list) or len(scores) != len(rows)):
        raise ValueError("V18 frozen input or score count differs")
    candidate, comparator = [], []
    for row, item in zip(rows, scores):
        baseline = item.get("baseline_scores")
        factor = item.get("factor_scores")
        if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                item.get("strict_parse") is not True or
                not isinstance(baseline, dict) or
                set(baseline) != set(applicable_codes(row["source_kind"])) or
                any(type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= 1 for value in baseline.values()) or
                (row["source_kind"] == "message" and
                 (not isinstance(factor, dict) or set(factor) != set(FIELDS))) or
                (row["source_kind"] != "message" and factor is not None) or
                (isinstance(factor, dict) and any(
                    type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= 1 for value in factor.values()))):
            raise ValueError("V18 saved score vector differs")
        candidate.append({**item, "predicted": logic["candidate_from_scores"](row, factor, baseline)})
        comparator.append({**item, "predicted": logic["baseline_from_scores"](row, baseline)})
    candidate_report = evaluate_pairs(rows, candidate)
    comparator_report = evaluate_pairs(rows, comparator)
    factor_report = logic["factor_metrics"](
        facts, [item for item in scores if item["factor_scores"] is not None])
    metrics = {entry["category"]: entry for entry in candidate_report["per_category"]}
    success = (candidate_report["exact_pairs"] >= 64 and
               metrics["D1"]["tp"] >= 42 and metrics["S1"]["tp"] >= 21 and
               candidate_report["false_review_on_negatives"] <= 6 and
               candidate_report["exact_pairs"] > comparator_report["exact_pairs"])
    if (run.get("candidate_predictions") != candidate or
            run.get("comparator_predictions") != comparator or
            run.get("candidate_report") != candidate_report or
            run.get("comparator_report") != comparator_report or
            run.get("factor_metrics") != factor_report or
            run.get("preregistered_success") != success):
        raise ValueError("V18 saved decisions or metrics differ")
    return {"v18_pairs": candidate_report["exact_pairs"],
            "v16_pairs": comparator_report["exact_pairs"],
            "v18_false_reviews": candidate_report["false_review_on_negatives"],
            "D1_tp": metrics["D1"]["tp"], "S1_tp": metrics["S1"]["tp"],
            "preregistered_success": success}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
