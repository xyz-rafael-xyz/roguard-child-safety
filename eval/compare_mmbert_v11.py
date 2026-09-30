"""Recompute the frozen v11 comparison against v10 and v9a."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256
from compare_qwen_balanced_v8 import pair_hits, paired_discordance

BATCH = "batch-0023"
RUNS = {
    "v11": ("ro-mmbert-v11-test-0023.json", "ro-mmbert-v11-dev-selection.json", "ro-mmbert-v11"),
    "v10": ("ro-mmbert-v10-test-0023.json", "ro-mmbert-v10-dev-selection.json", "ro-mmbert-v10"),
    "v9a": ("ro-mmbert-v9a-test-0023.json", "ro-mmbert-v9a-dev-selection.json", "ro-mmbert-v9a"),
}


def checked_encoder(root: Path, rows: list[dict], name: str) -> tuple[dict, dict, Path, Path]:
    run_name, choice_name, checkpoint = RUNS[name]
    run_dir = root / "eval/runs"
    run = json.loads((run_dir / run_name).read_text(encoding="utf-8"))
    choice_path = run_dir / choice_name
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    weights = root / "checkpoints" / checkpoint / f"epoch-{choice['selected_epoch']}" / "adapter_model.safetensors"
    threshold = choice["selected_threshold"]
    if (run.get("batches") != [BATCH] or run.get("selected_epoch") != choice["selected_epoch"] or
            run.get("selection_record_sha256") != sha256(choice_path) or
            run.get("selected_weight_sha256") != sha256(weights) or
            sha256(weights) != choice["selected_weight_sha256"] or
            run.get("test_sha256") != sha256(root / f"data/synthetic/{BATCH}.jsonl") or
            run.get("attestation_sha256") != sha256(root / f"data/synthetic/{BATCH}.review.json") or
            run.get("threshold") != threshold):
        raise ValueError(f"{name} run differs from selected study")
    if name == "v11" and (
        run.get("development_tag_amendment") != "stored_batch_0015_tag_means_attested_batch_0022_rows_v11_pretest" or
        run.get("pretest_amendment_sha256") != sha256(root / "docs/V11_PRETEST_AMENDMENT.md")
    ):
        raise ValueError("V11 pretest metadata amendment differs")
    if len(run.get("predictions", [])) != len(rows):
        raise ValueError(f"{name} prediction count differs")
    for row, item in zip(rows, run["predictions"]):
        scores = item.get("scores")
        codes = applicable_codes(row["source_kind"])
        if (not isinstance(scores, dict) or set(scores) != set(codes) or
                any(type(scores[code]) not in (int, float) or not math.isfinite(scores[code]) or
                    not 0 <= scores[code] <= 1 for code in codes) or
                item.get("predicted") != [code for code in codes if scores[code] >= threshold] or
                item.get("strict_parse") is not True):
            raise ValueError(f"Invalid {name} score or decision: {row['id']}")
    report = evaluate_pairs(rows, run["predictions"])
    if run.get("report") != report:
        raise ValueError(f"Stored {name} report differs")
    return run, report, choice_path, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior comparison")
    rows = load_approved(root, [BATCH])
    current, current_report, choice, weights = checked_encoder(root, rows, "v11")
    prior, prior_report, prior_choice, prior_weights = checked_encoder(root, rows, "v10")
    older, older_report, older_choice, older_weights = checked_encoder(root, rows, "v9a")
    run_dir = root / "eval/runs"
    success = (current_report["exact_pairs"] >= 58 and
               current_report["false_review_on_negatives"] <= 7 and
               all(item["tp"] >= 9 for item in current_report["per_category"]) and
               current_report["exact_pairs"] > prior_report["exact_pairs"] and
               current_report["exact_pairs"] > older_report["exact_pairs"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "frozen_v11_surface_comparison", "batch": BATCH,
        "test_sha256": sha256(root / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(root / f"data/synthetic/{BATCH}.review.json"),
        "selection_records_sha256": {"v11": sha256(choice), "v10": sha256(prior_choice),
                                     "v9a": sha256(older_choice)},
        "selected_weights_sha256": {"v11": sha256(weights), "v10": sha256(prior_weights),
                                    "v9a": sha256(older_weights)},
        "runs_sha256": {name: sha256(run_dir / name) for name in
                        (RUNS["v11"][0], RUNS["v10"][0], RUNS["v9a"][0])},
        "v11": current_report, "v10": prior_report, "v9a": older_report,
        "paired_discordance": {
            "v10": paired_discordance(pair_hits(rows, current), pair_hits(rows, prior)),
            "v9a": paired_discordance(pair_hits(rows, current), pair_hits(rows, older)),
        },
        "always_review_reference": {"exact_cards": 72, "exact_pairs": 0,
                                    "false_review_on_negatives": 72},
        "never_review_reference": {"exact_cards": 72, "exact_pairs": 0,
                                   "false_review_on_negatives": 0},
        "preregistered_success": success,
        "limitation": "Synthetic rule descriptions only; no authentic child-language validity.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
