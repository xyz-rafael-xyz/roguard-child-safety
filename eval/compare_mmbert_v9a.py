"""Recompute the frozen v9a adapter-versus-Qwen comparison on batch 0019."""

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

BATCH = "batch-0019"
ENCODER = "ro-mmbert-v9a-test-0019.json"
QWEN = "ro-qwen-v8-test-0019.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior comparison")
    rows = load_approved(root, [BATCH])
    run_dir = root / "eval/runs"
    encoder = json.loads((run_dir / ENCODER).read_text(encoding="utf-8"))
    qwen = json.loads((run_dir / QWEN).read_text(encoding="utf-8"))
    if encoder.get("batches") != [BATCH] or qwen.get("batches") != [BATCH]:
        raise ValueError("Comparison runs must use the same sealed test")
    choice = run_dir / "ro-mmbert-v9a-dev-selection.json"
    selected = json.loads(choice.read_text(encoding="utf-8"))
    weights = root / "checkpoints/ro-mmbert-v9a" / f"epoch-{selected['selected_epoch']}" / "adapter_model.safetensors"
    threshold = selected["selected_threshold"]
    if (encoder.get("selected_epoch") != selected["selected_epoch"] or
            encoder.get("selection_record_sha256") != sha256(choice) or
            encoder.get("selected_weight_sha256") != sha256(weights) or
            sha256(weights) != selected["selected_weight_sha256"] or
            encoder.get("test_sha256") != sha256(root / f"data/synthetic/{BATCH}.jsonl") or
            encoder.get("attestation_sha256") != sha256(root / f"data/synthetic/{BATCH}.review.json") or
            encoder.get("threshold") != threshold):
        raise ValueError("Encoder run differs from selected study")
    for row, item in zip(rows, encoder["predictions"]):
        scores = item.get("scores")
        codes = applicable_codes(row["source_kind"])
        if (not isinstance(scores, dict) or set(scores) != set(codes) or
                any(type(scores[code]) not in (int, float) or not math.isfinite(scores[code]) or
                    not 0 <= scores[code] <= 1 for code in codes) or
                item.get("predicted") != [code for code in codes if scores[code] >= threshold] or
                item.get("strict_parse") is not True):
            raise ValueError(f"Invalid encoder score or decision: {row['id']}")
    _, adapter = _selection(root, "ro-qwen-v8", "ro-qwen-v8-dev-selection.json")
    if (qwen.get("adapter") != str(adapter.relative_to(root)) or
            qwen.get("parser_profile") != PARSER_PROFILE or
            qwen.get("rows") != len(rows)):
        raise ValueError("Qwen comparator differs from selected v8 adapter")
    encoder_report = evaluate_pairs(rows, encoder["predictions"])
    qwen_report = evaluate_pairs(rows, qwen["predictions"])
    if encoder.get("report") != encoder_report:
        raise ValueError("Encoder's stored report differs from recomputation")
    success = (encoder_report["exact_pairs"] >= 58 and
               encoder_report["false_review_on_negatives"] <= 7 and
               all(item["tp"] >= 9 for item in encoder_report["per_category"]) and
               encoder_report["exact_pairs"] > qwen_report["exact_pairs"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "frozen_v9a_encoder_comparison", "batch": BATCH,
        "test_sha256": sha256(root / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(root / f"data/synthetic/{BATCH}.review.json"),
        "selection_record_sha256": sha256(choice),
        "selected_weight_sha256": sha256(weights),
        "runs_sha256": {name: sha256(run_dir / name) for name in (ENCODER, QWEN)},
        "encoder": encoder_report, "qwen_v8": qwen_report,
        "paired_discordance": paired_discordance(pair_hits(rows, encoder), pair_hits(rows, qwen)),
        "always_review_reference": {"exact_cards": 72, "exact_pairs": 0,
                                    "false_review_on_negatives": 72},
        "never_review_reference": {"exact_cards": 72, "exact_pairs": 0,
                                   "false_review_on_negatives": 0},
        "preregistered_success": success,
        "limitation": "Synthetic rule descriptions only; not a measure of authentic child-language detection.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
