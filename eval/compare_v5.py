"""Audit and compare the preregistered v5 held-out runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.prompt_v4 import applicable_codes
from roguard.review import CATEGORIES, load_approved, sha256, taxonomy_sha256

from compare_v4 import summarize

RUNS = {
    "token-margin-v5": "ro-v5-margin-test-0013.json",
    "hard-adapter-v4": "ro-v4-hard-test-0013.json",
    "romistral-base": "romistral-base-v4-0013.json",
}


def audit_margin(rows: list[dict], run: dict, calibration: dict) -> None:
    if (run.get("status") != "frozen_v5_token_margin_test" or
            run.get("batches") != ["batch-0013"] or
            run.get("thresholds") != calibration["thresholds"] or
            run.get("score_semantics") != calibration["score_semantics"] or
            len(run.get("predictions", ())) != len(rows)):
        raise ValueError("Margin run configuration differs")
    for row, item in zip(rows, run["predictions"]):
        if item["id"] != row["id"] or item["expected"] != row["labels"]:
            raise ValueError("Margin run rows differ")
        if set(item["scores"]) != set(CATEGORIES):
            raise ValueError("Margin run lacks a score")
        expected = tuple(code for code in applicable_codes(row["source_kind"])
                         if item["scores"][code] >= calibration["thresholds"][code])
        if tuple(item["predicted"]) != expected or item["strict_parse"] is not True:
            raise ValueError("Margin prediction differs from frozen thresholds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_approved(root, ["batch-0013"])
    if len(rows) != 96:
        raise ValueError("Frozen test length differs")
    run_dir = root / "eval" / "runs"
    calibration_path = run_dir / "ro-v5-dev-thresholds.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    loaded = {name: json.loads((run_dir / filename).read_text(encoding="utf-8"))
              for name, filename in RUNS.items()}
    margin = loaded["token-margin-v5"]
    audit_margin(rows, margin, calibration)
    if (margin["test_sha256"] != sha256(root / "data" / "synthetic" / "batch-0013.jsonl") or
            margin["attestation_sha256"] != sha256(root / "data" / "synthetic" / "batch-0013.review.json") or
            margin["threshold_record_sha256"] != sha256(calibration_path) or
            margin["prompt_sha256"] != sha256(root / "src" / "roguard" / "prompt_v4.py") or
            margin["adapter_weight_sha256"] != sha256(root / "checkpoints" / "ro-v4" / "selected-adapter" / "adapters.safetensors")):
        raise ValueError("Margin run provenance differs")
    hard = loaded["hard-adapter-v4"]
    if hard.get("batches") != ["batch-0013"] or hard.get("rows") != 96:
        raise ValueError("Hard adapter run configuration differs")
    base = loaded["romistral-base"]
    registry = json.loads((root / "eval" / "models.json").read_text(encoding="utf-8"))["romistral7b"]
    if (base.get("batches") != ["batch-0013"] or base.get("model") != registry["model"] or
            base.get("revision") != registry["revision"] or
            base.get("prompt_version") != "v4_binary" or base.get("max_tokens_per_task") != 8 or
            base.get("taxonomy_sha256") != {"ro": taxonomy_sha256(root, "ro")}):
        raise ValueError("Prompt-only base configuration differs")
    reports = [summarize(rows, margin, "ro-v4-adapter"),
               summarize(rows, hard, "ro-v4-adapter"),
               summarize(rows, base, "romistral-base")]
    for report, name in zip(reports, RUNS):
        report["model"] = name
    margin_report, hard_report, base_report = reports
    category_gate = all(item["tp"] >= 4 and item["fp"] <= 2
                        for item in margin_report["per_category"])
    result = {
        "status": "frozen_v5_comparison", "test_batch": "batch-0013",
        "test_sha256": sha256(root / "data" / "synthetic" / "batch-0013.jsonl"),
        "attestation_sha256": sha256(root / "data" / "synthetic" / "batch-0013.review.json"),
        "threshold_record_sha256": sha256(calibration_path),
        "run_sha256": {filename: sha256(run_dir / filename) for filename in RUNS.values()},
        "reports": reports,
        "always_review_reference": {"exact_rows": 48, "exact_pairs": 0,
                                    "correct_direction_flips": 0, "false_review_on_negatives": 48},
        "never_review_reference": {"exact_rows": 48, "exact_pairs": 0,
                                   "correct_direction_flips": 0, "false_review_on_negatives": 0},
        "preregistered_success": (margin_report["exact_pairs"] > hard_report["exact_pairs"] and
                                  margin_report["exact_pairs"] > base_report["exact_pairs"] and category_gate),
        "limitation": "Symbolic policy descriptions only; no authentic child language or deployed child-safety validity.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Comparison already exists")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
