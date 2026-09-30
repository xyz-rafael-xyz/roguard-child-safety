"""Recompute v12 reports and threshold decisions from saved test scores."""

from __future__ import annotations

import json
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.review import CATEGORIES, load_approved, sha256
from compare_qwen_balanced_v8 import pair_hits, paired_discordance
from select_mmbert_v12 import OUTPUT as CHOICE_PATH, V11_CHOICE, apply_thresholds

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v12-test-0024.json")


def verify(root: Path = ROOT) -> dict:
    choice = json.loads((root / CHOICE_PATH).read_text(encoding="utf-8"))
    v11 = json.loads((root / V11_CHOICE).read_text(encoding="utf-8"))
    run = json.loads((root / RUN).read_text(encoding="utf-8"))
    rows = load_approved(root, ["batch-0024"])
    if (run.get("batch") != "batch-0024" or
            run.get("test_sha256") != sha256(root / "data/synthetic/batch-0024.jsonl") or
            run.get("attestation_sha256") != sha256(root / "data/synthetic/batch-0024.review.json") or
            run.get("selection_record_sha256") != sha256(root / CHOICE_PATH) or
            run.get("selected_weight_sha256") != choice["selected_weight_sha256"] or
            run.get("thresholds") != choice["thresholds"] or
            run.get("v11_shared_threshold") != v11["selected_threshold"]):
        raise ValueError("V12 run, selection, or attestation differs")
    current = run["predictions"]
    prior = run["v11_predictions"]
    if len(current) != len(rows) or len(prior) != len(rows):
        raise ValueError("V12 prediction count differs")
    for row, item, baseline in zip(rows, current, prior):
        if (item.get("id") != row["id"] or baseline.get("id") != row["id"] or
                item.get("scores") != baseline.get("scores") or
                item.get("strict_parse") is not True or baseline.get("strict_parse") is not True):
            raise ValueError("Raw score or row order differs")
    raw = [{**item, "predicted": []} for item in current]
    if (apply_thresholds(raw, choice["thresholds"]) != current or
            apply_thresholds(raw, {code: v11["selected_threshold"] for code in CATEGORIES}) != prior):
        raise ValueError("Stored labels do not match the frozen score thresholds")
    report = evaluate_pairs(rows, current)
    baseline = evaluate_pairs(rows, prior)
    discordance = paired_discordance(pair_hits(rows, {"predictions": current}),
                                    pair_hits(rows, {"predictions": prior}))
    success = (report["exact_pairs"] >= 58 and report["false_review_on_negatives"] <= 7 and
               all(item["tp"] >= 9 for item in report["per_category"]) and
               report["exact_pairs"] > baseline["exact_pairs"])
    if (run["report"] != report or run["v11_report"] != baseline or
            run["paired_discordance"] != discordance or run["preregistered_success"] != success):
        raise ValueError("Stored V12 metrics differ from saved predictions")
    return {"v12_pairs": report["exact_pairs"], "v11_pairs": baseline["exact_pairs"],
            "v12_false_reviews": report["false_review_on_negatives"],
            "preregistered_success": success, "paired_discordance": discordance}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
