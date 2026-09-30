"""Recompute the prospective V17 cutoff study from frozen score vectors."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v17-test-0032.json")
MODEL = Path("models/ro-mmbert-v16-abstract")


def verify(root: Path = ROOT) -> dict:
    study = runpy.run_path(str(root / "eval/run_mmbert_v17.py"))
    selection = study["verify_selection"]()
    run = json.loads((root / RUN).read_text(encoding="utf-8"))
    rows = load_approved(root, ["batch-0032"])
    scores = run.get("scores")
    if (run.get("status") != "frozen_v17_cutoff_test" or
            run.get("batch") != "batch-0032" or len(rows) != 96 or
            run.get("batch_sha256") != sha256(root / "data/synthetic/batch-0032.jsonl") or
            run.get("attestation_sha256") != sha256(root / "data/synthetic/batch-0032.review.json") or
            run.get("protocol_sha256") != sha256(root / "docs/V17_PROTOCOL.md") or
            run.get("runner_sha256") != sha256(root / "eval/run_mmbert_v17.py") or
            run.get("selection_sha256") != sha256(root / "eval/runs/ro-mmbert-v17-dev-selection.json") or
            run.get("model_manifest_sha256") != sha256(root / MODEL / "research.json") or
            run.get("model_weight_sha256") != sha256(root / MODEL / "adapter_model.safetensors") or
            run.get("d1_cutoff") != selection["d1_cutoff"] or
            run.get("s1_cutoff") != 0.5 or
            not isinstance(scores, list) or len(scores) != len(rows)):
        raise ValueError("V17 frozen input or score count differs")
    for row, item in zip(rows, scores):
        if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                item.get("strict_parse") is not True or
                set(item.get("scores", {})) != set(applicable_codes(row["source_kind"])) or
                any(type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= 1 for value in item["scores"].values())):
            raise ValueError("V17 saved test scores differ")
    candidate = study["decisions"](rows, scores, selection["d1_cutoff"])
    comparator = study["decisions"](rows, scores, 0.5)
    candidate_report = evaluate_pairs(rows, candidate)
    comparator_report = evaluate_pairs(rows, comparator)
    categories = {item["category"]: item for item in candidate_report["per_category"]}
    success = (selection["development_recall_floor_met"] and
               candidate_report["exact_pairs"] >= 42 and
               candidate_report["false_review_on_negatives"] <= 4 and
               all(categories[code]["tp"] >= 21 for code in ("D1", "S1")) and
               candidate_report["exact_pairs"] > comparator_report["exact_pairs"])
    if (run.get("candidate_predictions") != candidate or
            run.get("comparator_predictions") != comparator or
            run.get("candidate_report") != candidate_report or
            run.get("comparator_report") != comparator_report or
            run.get("preregistered_success") != success):
        raise ValueError("V17 saved decisions or metrics differ")
    return {"v17_pairs": candidate_report["exact_pairs"],
            "v16_pairs": comparator_report["exact_pairs"],
            "v17_false_reviews": candidate_report["false_review_on_negatives"],
            "D1_tp": categories["D1"]["tp"], "S1_tp": categories["S1"]["tp"],
            "preregistered_success": success}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
