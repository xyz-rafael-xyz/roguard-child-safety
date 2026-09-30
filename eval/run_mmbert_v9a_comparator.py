"""Evaluate the unchanged v9a adapter on the new v10 holdout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roguard.mmbert_study import (BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION,
                                  predict_rows, verify_base)
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
CHOICE = Path("eval/runs/ro-mmbert-v9a-dev-selection.json")
BATCH = "batch-0020"


def committed_choice(root: Path) -> dict:
    saved = subprocess.run(["git", "show", f"HEAD:{CHOICE.as_posix()}"], cwd=root,
                           check=True, capture_output=True).stdout
    if (root / CHOICE).read_bytes() != saved:
        raise ValueError("Commit the development selection before test inference")
    return json.loads(saved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior v9a test run")
    choice = committed_choice(root)
    base = args.base_model_path.resolve()
    verify_base(base)
    if (choice.get("amendment") != "v9a_pairwise_lora_pretest" or
            choice.get("model") != MODEL_ID or choice.get("revision") != REVISION or
            choice.get("base_files_sha256") != BASE_HASHES or
            choice.get("train_batch") != "batch-0014" or
            choice.get("development_batch") != "batch-0015" or
            choice.get("train_batch_sha256") != sha256(root / "data/synthetic/batch-0014.jsonl") or
            choice.get("development_sha256") != sha256(root / "data/synthetic/batch-0015.jsonl") or
            choice.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            choice.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py") or
            choice.get("study_code_sha256") != sha256(root / "src/roguard/mmbert_study.py") or
            choice.get("trainer_sha256") != sha256(root / "training/train_mmbert_pairs.py") or
            choice.get("protocol_sha256") != sha256(root / "docs/V9A_AMENDMENT.md") or
            choice.get("max_length") != MAX_LENGTH or choice.get("total_steps") != 1080 or
            choice.get("pair_exposures_per_epoch") != 864):
        raise ValueError("V9a model or protocol differs from selected study")
    if len(choice.get("candidates", [])) != 5 or sorted(c["epoch"] for c in choice["candidates"]) != [1, 2, 3, 4, 5]:
        raise ValueError("V9a did not evaluate all registered epochs")
    best = max(choice["candidates"], key=lambda c: (
        c["report"]["exact_pairs"], c["report"]["exact_cards"],
        -c["report"]["false_review_on_negatives"],
        -abs(c["threshold"] - 0.5), -c["epoch"], -c["threshold"]))
    epoch, threshold = best["epoch"], best["threshold"]
    if epoch != choice.get("selected_epoch") or threshold != choice.get("selected_threshold"):
        raise ValueError("Selected epoch or threshold differs from development rule")
    adapter = root / "checkpoints/ro-mmbert-v9a" / f"epoch-{epoch}"
    weights = adapter / "adapter_model.safetensors"
    dev_path = root / "checkpoints/ro-mmbert-v9a" / f"epoch-{epoch}-dev.json"
    if (best["weight_sha256"] != sha256(weights) or
            choice.get("selected_weight_sha256") != sha256(weights) or
            best["dev_predictions_sha256"] != sha256(dev_path)):
        raise ValueError("Selected adapter or development predictions differ")
    dev_rows = load_approved(root, ["batch-0015"])
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if (dev.get("threshold") != threshold or dev.get("batches") != ["batch-0015"] or
            evaluate_pairs(dev_rows, dev["predictions"]) != best["report"]):
        raise ValueError("Selected development metrics differ")
    rows = load_approved(root, [BATCH])
    if len(rows) != 144 or any(row["split"] != "test" for row in rows):
        raise ValueError("Frozen v9a test batch differs")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=False, local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).to(device)
    items = predict_rows(model, tokenizer, rows, device, threshold=threshold)
    report = evaluate_pairs(rows, items)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "frozen_v9a_comparator_run", "batches": [BATCH],
        "test_sha256": sha256(root / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(root / f"data/synthetic/{BATCH}.review.json"),
        "selection_record_sha256": sha256(root / CHOICE),
        "selected_epoch": epoch, "selected_weight_sha256": sha256(weights),
        "model": MODEL_ID, "revision": REVISION, "threshold": threshold,
        "score_semantics": "development_threshold_on_uncalibrated_symbolic_softmax",
        "predictions": items, "report": report,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
