"""Recompute the frozen failed V15 advisory prose result from saved scores."""

from __future__ import annotations

import json
import math
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v15-test-0027.json")
BATCH = "batch-0027"


def verify(root: Path = ROOT) -> dict:
    run = json.loads((root / RUN).read_text(encoding="utf-8"))
    manifest_path = root / "models/ro-mmbert-v11-abstract/research.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_approved(root, [BATCH])
    if (run.get("status") != "frozen_v15_advisory_prose_test" or
            run.get("batch") != BATCH or len(rows) != 96 or
            run.get("test_sha256") != sha256(root / f"data/synthetic/{BATCH}.jsonl") or
            run.get("attestation_sha256") != sha256(root / f"data/synthetic/{BATCH}.review.json") or
            run.get("protocol_sha256") != sha256(root / "docs/V15_PROTOCOL.md") or
            run.get("adapter_weight_sha256") != manifest["adapter_weight_sha256"] or
            run.get("adapter_weight_sha256") != sha256(root / "models/ro-mmbert-v11-abstract/adapter_model.safetensors") or
            run.get("artifact_manifest_sha256") != sha256(manifest_path) or
            run.get("threshold") != manifest["shared_threshold"]):
        raise ValueError("V15 frozen inputs or model artifact differ")
    items = run.get("predictions")
    if not isinstance(items, list) or len(items) != len(rows):
        raise ValueError("V15 prediction count differs")
    for row, item in zip(rows, items):
        scores = item.get("scores")
        codes = applicable_codes(row["source_kind"])
        if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                item.get("strict_parse") is not True or
                not isinstance(scores, dict) or set(scores) != set(codes) or
                any(type(scores[code]) not in (int, float) or
                    not math.isfinite(scores[code]) or not 0 <= scores[code] <= 1
                    for code in codes) or
                item.get("predicted") != [code for code in codes
                                          if scores[code] >= run["threshold"]]):
            raise ValueError(f"Invalid V15 score or decision: {row['id']}")
    report = evaluate_pairs(rows, items)
    by_code = {entry["category"]: entry for entry in report["per_category"]}
    success = (report["exact_pairs"] >= 42 and
               report["false_review_on_negatives"] <= 4 and
               all(by_code[code]["tp"] >= 21 for code in ("D1", "S1")))
    always = [{"id": row["id"], "predicted": ["D1" if row["source_kind"] == "message" else "S1"],
               "strict_parse": True} for row in rows]
    never = [{"id": row["id"], "predicted": [], "strict_parse": True} for row in rows]
    for name, predictions in (("always_review_reference", always), ("never_review_reference", never)):
        expected = evaluate_pairs(rows, predictions)
        if any(run[name][key] != expected[key] for key in
               ("exact_cards", "exact_pairs", "false_review_on_negatives")):
            raise ValueError(f"V15 {name} differs")
    if run.get("report") != report or run.get("preregistered_success") != success:
        raise ValueError("V15 stored metrics differ from predictions")
    return {"exact_pairs": report["exact_pairs"],
            "false_reviews": report["false_review_on_negatives"],
            "D1_tp": by_code["D1"]["tp"], "S1_tp": by_code["S1"]["tp"],
            "preregistered_success": success}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
