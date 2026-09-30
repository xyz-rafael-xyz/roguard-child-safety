"""Recompute the frozen failed V16 comparison from saved score vectors."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v16-test-0030.json")
BATCH = "batch-0030"
V16 = Path("models/ro-mmbert-v16-abstract")
V11 = Path("models/ro-mmbert-v11-abstract")


def verify(root: Path = ROOT) -> dict:
    run = json.loads((root / RUN).read_text(encoding="utf-8"))
    v16 = json.loads((root / V16 / "research.json").read_text(encoding="utf-8"))
    v11 = json.loads((root / V11 / "research.json").read_text(encoding="utf-8"))
    selected = runpy.run_path(str(root / "eval/verify_committed_v16_selection.py"))[
        "verify"](root)
    rows = load_approved(root, [BATCH])
    if (run.get("status") != "frozen_v16_advisory_test" or
            run.get("batch") != BATCH or len(rows) != 96 or
            run.get("test_sha256") != sha256(root / f"data/synthetic/{BATCH}.jsonl") or
            run.get("attestation_sha256") != sha256(root / f"data/synthetic/{BATCH}.review.json") or
            run.get("protocol_sha256") != sha256(root / "docs/V16_PROTOCOL.md") or
            run.get("prefit_amendment_sha256") != sha256(root / "docs/V16_PREFIT_AMENDMENT.md") or
            run.get("selection_record_sha256") != sha256(root / "eval/runs/ro-mmbert-v16-dev-selection.json") or
            run.get("v16_manifest_sha256") != sha256(root / V16 / "research.json") or
            run.get("v11_manifest_sha256") != sha256(root / V11 / "research.json") or
            run.get("v16_weight_sha256") != v16["adapter_weight_sha256"] or
            run.get("v11_weight_sha256") != v11["adapter_weight_sha256"] or
            run.get("v16_threshold") != v16["shared_threshold"] or
            run.get("v11_threshold") != v11["shared_threshold"] or
            run.get("selected_development") != selected):
        raise ValueError("V16 frozen inputs, selection, or weights differ")
    candidate = run.get("candidate_predictions")
    comparator = run.get("comparator_predictions")
    if (not isinstance(candidate, list) or not isinstance(comparator, list) or
            len(candidate) != len(rows) or len(comparator) != len(rows)):
        raise ValueError("V16 prediction count differs")
    for predictions, threshold in ((candidate, run["v16_threshold"]),
                                   (comparator, run["v11_threshold"])):
        for row, item in zip(rows, predictions):
            scores = item.get("scores")
            codes = applicable_codes(row["source_kind"])
            if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                    item.get("strict_parse") is not True or
                    not isinstance(scores, dict) or set(scores) != set(codes) or
                    any(type(scores[code]) not in (int, float) or
                        not math.isfinite(scores[code]) or not 0 <= scores[code] <= 1
                        for code in codes) or
                    item.get("predicted") != [code for code in codes
                                              if scores[code] >= threshold]):
                raise ValueError(f"Invalid V16 saved score or decision: {row['id']}")
    candidate_report = evaluate_pairs(rows, candidate)
    comparator_report = evaluate_pairs(rows, comparator)
    by_code = {entry["category"]: entry for entry in candidate_report["per_category"]}
    success = (candidate_report["exact_pairs"] >= 42 and
               candidate_report["false_review_on_negatives"] <= 4 and
               all(by_code[code]["tp"] >= 21 for code in ("D1", "S1")) and
               candidate_report["exact_pairs"] > comparator_report["exact_pairs"])
    logic = runpy.run_path(str(root / "eval/run_mmbert_v16.py"))
    always = [{"id": row["id"], "predicted": ["D1" if row["source_kind"] == "message" else "S1"],
               "strict_parse": True} for row in rows]
    never = [{"id": row["id"], "predicted": [], "strict_parse": True} for row in rows]
    if (run.get("candidate_report") != candidate_report or
            run.get("comparator_report") != comparator_report or
            run.get("paired_discordance") != logic["discordance"](rows, candidate, comparator) or
            run.get("always_review_reference") != evaluate_pairs(rows, always) or
            run.get("never_review_reference") != evaluate_pairs(rows, never) or
            run.get("preregistered_success") != success):
        raise ValueError("V16 stored metrics differ from saved decisions")
    return {"v16_exact_pairs": candidate_report["exact_pairs"],
            "v11_exact_pairs": comparator_report["exact_pairs"],
            "v16_false_reviews": candidate_report["false_review_on_negatives"],
            "D1_tp": by_code["D1"]["tp"], "S1_tp": by_code["S1"]["tp"],
            "preregistered_success": success}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
