"""Fit and select the preregistered v9 Romanian symbolic-card encoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from roguard.mmbert_study import (BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION,
                                  THRESHOLD, predict_rows, verify_base)
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_train_dev, sha256, taxonomy_sha256
from roguard.prompt_v4 import applicable_codes
from prepare_balanced_mlx import prepare_balanced

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260930
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "ro-mmbert-v9")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("V9 output already exists; preserve prior study")
    verify_base(base)
    output.mkdir(parents=True)
    export = prepare_balanced(ROOT, output / "data")
    train_rows, dev_rows = load_train_dev(ROOT, ["batch-0014", "batch-0015"], "ro")
    if len(train_rows) != 576 or len(dev_rows) != 96 or export["train_tasks"] != 1728:
        raise ValueError("Unexpected v9 training/development data")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=False, local_files_only=True).to(device)
    train = [json.loads(line) for line in (output / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    encoded = tokenizer([task["prompt"] for task in train], truncation=True, max_length=MAX_LENGTH)
    examples = [{**{key: encoded[key][index] for key in encoded},
                 "labels": 1 if task["completion"] == "da" else 0}
                for index, task in enumerate(train)]
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS * len(loader)
    warmup = max(1, round(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup
        else max(0.0, (total_steps - step) / (total_steps - warmup)))
    record = {"status": "development_selection", "model": MODEL_ID, "revision": REVISION,
              "base_files_sha256": BASE_HASHES, "train_batch": "batch-0014",
              "development_batch": "batch-0015",
              "train_batch_sha256": sha256(ROOT / "data/synthetic/batch-0014.jsonl"),
              "development_sha256": sha256(ROOT / "data/synthetic/batch-0015.jsonl"),
              "train_export_sha256": export["train_sha256"],
              "dev_export_sha256": export["valid_sha256"],
              "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
              "prompt_sha256": sha256(ROOT / "src/roguard/prompt_v4.py"),
              "study_code_sha256": sha256(ROOT / "src/roguard/mmbert_study.py"),
              "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
              "learning_rate": LR, "max_length": MAX_LENGTH, "threshold": THRESHOLD,
              "warmup_steps": warmup, "total_steps": total_steps,
              "device": device, "torch_version": torch.__version__,
              "transformers_version": __import__("transformers").__version__, "candidates": []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            if step % 36 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} mean_loss={loss_sum / step:.4f}", flush=True)
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        items = predict_rows(model, tokenizer, dev_rows, device)
        report = evaluate_pairs(dev_rows, items)
        (output / f"epoch-{epoch}-dev.json").write_text(
            json.dumps({"batches": ["batch-0015"], "predictions": items,
                        "report": report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        candidate = {"epoch": epoch, "mean_train_loss": loss_sum / len(loader),
                     "weight_sha256": sha256(checkpoint / "model.safetensors"),
                     "dev_predictions_sha256": sha256(output / f"epoch-{epoch}-dev.json"),
                     "report": report}
        record["candidates"].append(candidate)
        print(f"epoch={epoch} dev_pairs={report['exact_pairs']}/48 dev_cards={report['exact_cards']}/96", flush=True)
    best = max(record["candidates"], key=lambda c: (
        c["report"]["exact_pairs"], c["report"]["exact_cards"],
        -c["report"]["false_review_on_negatives"], -c["epoch"]))
    record["selected_epoch"] = best["epoch"]
    record["selected_weight_sha256"] = best["weight_sha256"]
    target = output / "development-selection.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected={best['epoch']} record={target}", flush=True)


if __name__ == "__main__":
    main()
