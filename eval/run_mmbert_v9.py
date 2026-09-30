"""Run the selected v9 encoder once on its sealed test batch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roguard.mmbert_study import (BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION,
                                  THRESHOLD, predict_rows, verify_base)
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
CHOICE = Path("eval/runs/ro-mmbert-v9-dev-selection.json")
BATCH = "batch-0019"


def committed_choice(root: Path) -> dict:
    path = root / CHOICE
    saved = subprocess.run(["git", "show", f"HEAD:{CHOICE.as_posix()}"], cwd=root,
                           check=True, capture_output=True).stdout
    if path.read_bytes() != saved:
        raise ValueError("Development selection must be committed before test inference")
    return json.loads(saved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior v9 test run")
    choice = committed_choice(root)
    base = args.base_model_path.resolve()
    verify_base(base)
    if (choice.get("model") != MODEL_ID or choice.get("revision") != REVISION or
            choice.get("base_files_sha256") != BASE_HASHES or
            choice.get("train_batch") != "batch-0014" or
            choice.get("development_batch") != "batch-0015" or
            choice.get("train_batch_sha256") != sha256(root / "data/synthetic/batch-0014.jsonl") or
            choice.get("development_sha256") != sha256(root / "data/synthetic/batch-0015.jsonl") or
            choice.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            choice.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py") or
            choice.get("study_code_sha256") != sha256(root / "src/roguard/mmbert_study.py") or
            choice.get("threshold") != THRESHOLD or choice.get("max_length") != MAX_LENGTH):
        raise ValueError("Selected encoder or study inputs differ")
    epoch = choice["selected_epoch"]
    best = max(choice["candidates"], key=lambda c: (
        c["report"]["exact_pairs"], c["report"]["exact_cards"],
        -c["report"]["false_review_on_negatives"], -c["epoch"]))
    if epoch != best["epoch"] or len(choice["candidates"]) != 3 or choice.get("total_steps") != 648:
        raise ValueError("Development choice differs from the fixed three-epoch rule")
    candidate = next(item for item in choice["candidates"] if item["epoch"] == epoch)
    weights = root / "checkpoints" / "ro-mmbert-v9" / f"epoch-{epoch}" / "model.safetensors"
    if sha256(weights) != choice["selected_weight_sha256"] or candidate["weight_sha256"] != choice["selected_weight_sha256"]:
        raise ValueError("Selected encoder weight differs")
    train_export = root / "checkpoints/ro-mmbert-v9/data/train.jsonl"
    dev_export = root / "checkpoints/ro-mmbert-v9/data/valid.jsonl"
    dev_predictions = root / "checkpoints/ro-mmbert-v9" / f"epoch-{epoch}-dev.json"
    if (choice["train_export_sha256"] != sha256(train_export) or
            choice["dev_export_sha256"] != sha256(dev_export) or
            candidate["dev_predictions_sha256"] != sha256(dev_predictions)):
        raise ValueError("Development export or predictions differ from selection")
    rows = load_approved(root, [BATCH])
    if len(rows) != 144 or any(row["split"] != "test" for row in rows):
        raise ValueError("Frozen v9 test batch differs")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(weights.parent, local_files_only=True).to(device)
    items = predict_rows(model, tokenizer, rows, device)
    report = evaluate_pairs(rows, items)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "frozen_v9_test_run", "batches": [BATCH],
        "test_sha256": sha256(root / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(root / f"data/synthetic/{BATCH}.review.json"),
        "selection_record_sha256": sha256(root / CHOICE),
        "selected_epoch": epoch, "selected_weight_sha256": sha256(weights),
        "model": MODEL_ID, "revision": REVISION, "threshold": THRESHOLD,
        "score_semantics": "uncalibrated_softmax_class_one_on_symbolic_cards",
        "predictions": items, "report": report,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
