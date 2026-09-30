"""One frozen v12 inference on a fresh abstract Romanian surface."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roguard.mmbert_study import (BASE_HASHES, MODEL_ID, REVISION, predict_rows,
                                  verify_base)
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256
from compare_qwen_balanced_v8 import pair_hits, paired_discordance
from select_mmbert_v12 import (DEV_SCORES, OUTPUT as CHOICE_PATH, V11_CHOICE,
                               apply_thresholds, select_thresholds,
                               validated_inputs)

ROOT = Path(__file__).resolve().parents[1]
BATCH = "batch-0024"


def committed_choice(root: Path) -> dict:
    saved = subprocess.run(["git", "show", f"HEAD:{CHOICE_PATH.as_posix()}"], cwd=root,
                           check=True, capture_output=True).stdout
    if (root / CHOICE_PATH).read_bytes() != saved:
        raise ValueError("Commit the development selection before test inference")
    return json.loads(saved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = ROOT.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior v12 test run")
    choice = committed_choice(root)
    v11 = json.loads((root / V11_CHOICE).read_text(encoding="utf-8"))
    rows_dev, items_dev, _ = validated_inputs(root)
    thresholds, dev_report, counts = select_thresholds(rows_dev, items_dev)
    if (choice.get("study") != "v12_category_threshold_transfer" or
            choice.get("fit") != "unchanged_v11_epoch_3" or
            choice.get("model") != MODEL_ID or choice.get("revision") != REVISION or
            choice.get("selected_weight_sha256") != v11["selected_weight_sha256"] or
            choice.get("v11_selection_sha256") != sha256(root / V11_CHOICE) or
            choice.get("development_batch") != "batch-0022" or
            choice.get("development_sha256") != sha256(root / "data/synthetic/batch-0022.jsonl") or
            choice.get("development_scores_sha256") != sha256(root / DEV_SCORES) or
            choice.get("protocol_sha256") != sha256(root / "docs/V12_PROTOCOL.md") or
            choice.get("selector_sha256") != sha256(root / "eval/select_mmbert_v12.py") or
            choice.get("sealed_test_batch") != BATCH or
            choice.get("sealed_test_sha256") != sha256(root / f"data/synthetic/{BATCH}.jsonl") or
            choice.get("thresholds") != thresholds or
            choice.get("candidate_counts") != counts or
            choice.get("development_report") != dev_report):
        raise ValueError("V12 selection, code, or sealed test differs")
    base = args.base_model_path.resolve()
    verify_base(base)
    if v11.get("base_files_sha256") != BASE_HASHES:
        raise ValueError("V11 selected a different base")
    adapter = root / "checkpoints/ro-mmbert-v11/epoch-3"
    weights = adapter / "adapter_model.safetensors"
    if sha256(weights) != choice["selected_weight_sha256"]:
        raise ValueError("V12 adapter differs from the selected v11 weights")
    rows = load_approved(root, [BATCH])
    if len(rows) != 144 or any(row["split"] != "test" for row in rows):
        raise ValueError("V12 test batch differs")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=False, local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).to(device)
    raw = predict_rows(model, tokenizer, rows, device)
    proposed = apply_thresholds(raw, thresholds)
    prior = apply_thresholds(raw, {code: v11["selected_threshold"] for code in thresholds})
    current_report = evaluate_pairs(rows, proposed)
    prior_report = evaluate_pairs(rows, prior)
    success = (current_report["exact_pairs"] >= 58 and
               current_report["false_review_on_negatives"] <= 7 and
               all(item["tp"] >= 9 for item in current_report["per_category"]) and
               current_report["exact_pairs"] > prior_report["exact_pairs"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "frozen_v12_category_threshold_test", "batch": BATCH,
        "test_sha256": sha256(root / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(root / f"data/synthetic/{BATCH}.review.json"),
        "selection_record_sha256": sha256(root / CHOICE_PATH),
        "selected_weight_sha256": sha256(weights),
        "base_revision": REVISION, "thresholds": thresholds,
        "v11_shared_threshold": v11["selected_threshold"],
        "score_semantics": "uncalibrated_symbolic_softmax",
        "predictions": proposed, "v11_predictions": prior,
        "report": current_report, "v11_report": prior_report,
        "paired_discordance": paired_discordance(
            pair_hits(rows, {"predictions": proposed}), pair_hits(rows, {"predictions": prior})),
        "preregistered_success": success,
        "limitation": "Synthetic rule cards only; no authentic child-language validity.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"v12": current_report, "v11": prior_report,
                      "preregistered_success": success}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
