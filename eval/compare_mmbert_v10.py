"""Recompute the frozen v10 comparison against v9a and v8 Qwen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.qwen_output import PARSER_PROFILE
from roguard.review import load_approved, sha256
from compare_qwen_balanced_v8 import _selection, pair_hits, paired_discordance

BATCH = "batch-0020"
RUNS = {
    "v10": ("ro-mmbert-v10-test-0020.json", "ro-mmbert-v10-dev-selection.json", "ro-mmbert-v10"),
    "v9a": ("ro-mmbert-v9a-test-0020.json", "ro-mmbert-v9a-dev-selection.json", "ro-mmbert-v9a"),
}
QWEN = "ro-qwen-v8-test-0020.json"


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
    current, current_report, choice, weights = checked_encoder(root, rows, "v10")
    prior, prior_report, prior_choice, prior_weights = checked_encoder(root, rows, "v9a")
    run_dir = root / "eval/runs"
    qwen = json.loads((run_dir / QWEN).read_text(encoding="utf-8"))
    _, adapter = _selection(root, "ro-qwen-v8", "ro-qwen-v8-dev-selection.json")
    if (qwen.get("batches") != [BATCH] or
            qwen.get("adapter") != str(adapter.relative_to(root)) or
            qwen.get("parser_profile") != PARSER_PROFILE or
            qwen.get("rows") != len(rows)):
        raise ValueError("Qwen comparator differs from selected v8 adapter")
    qwen_report = evaluate_pairs(rows, qwen["predictions"])
    success = (current_report["exact_pairs"] >= 58 and
               current_report["false_review_on_negatives"] <= 7 and
               all(item["tp"] >= 9 for item in current_report["per_category"]) and
               current_report["exact_pairs"] > prior_report["exact_pairs"] and
               current_report["exact_pairs"] > qwen_report["exact_pairs"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "frozen_v10_cross_category_comparison", "batch": BATCH,
        "test_sha256": sha256(root / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(root / f"data/synthetic/{BATCH}.review.json"),
        "selection_records_sha256": {"v10": sha256(choice), "v9a": sha256(prior_choice)},
        "selected_weights_sha256": {"v10": sha256(weights), "v9a": sha256(prior_weights)},
        "runs_sha256": {name: sha256(run_dir / name) for name in
                        (RUNS["v10"][0], RUNS["v9a"][0], QWEN)},
        "v10": current_report, "v9a": prior_report, "qwen_v8": qwen_report,
        "paired_discordance": {
            "v9a": paired_discordance(pair_hits(rows, current), pair_hits(rows, prior)),
            "qwen_v8": paired_discordance(pair_hits(rows, current), pair_hits(rows, qwen)),
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
