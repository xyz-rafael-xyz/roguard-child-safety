"""Fit the preregistered v9a pairwise LoRA encoder on abstract cards."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roguard.mmbert_study import (BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION,
                                  predict_rows, verify_base)
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import make_prompt_v4
from roguard.review import CATEGORIES, load_train_dev, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260930
EPOCHS = 5
PAIRS_PER_BATCH = 4
LR = 1e-4
PAIR_LOSS_WEIGHT = 0.5
REPEAT = {code: 3 for code in CATEGORIES}


def choices_from_scores(rows: list[dict], items: list[dict]) -> tuple[float, dict, list[dict], int]:
    """Choose one shared development cutoff by the prospective pair-first rule."""
    scores = sorted({score for item in items for score in item["scores"].values()})
    thresholds = sorted({0.0, 0.5, 1.0} | {(left + right) / 2 for left, right in zip(scores, scores[1:])})
    best = None
    for threshold in thresholds:
        predicted = [{**item, "predicted": [code for code, score in item["scores"].items()
                                           if score >= threshold]} for item in items]
        report = evaluate_pairs(rows, predicted)
        key = (report["exact_pairs"], report["exact_cards"],
               -report["false_review_on_negatives"], -abs(threshold - 0.5), -threshold)
        if best is None or key > best[0]:
            best = (key, threshold, report, predicted)
    assert best is not None
    return best[1], best[2], best[3], len(thresholds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/ro-mmbert-v9a")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("V9a output already exists; preserve prior run")
    verify_base(base)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit study code and protocol before fitting")
    train_rows, dev_rows = load_train_dev(ROOT, ["batch-0014", "batch-0015"], "ro")
    if len(train_rows) != 576 or len(dev_rows) != 96:
        raise ValueError("Unexpected training/development cardinality")
    pairs = []
    counts = {code: 0 for code in CATEGORIES}
    for negative, positive in zip(train_rows[::2], train_rows[1::2]):
        if negative["labels"] or len(positive["labels"]) != 1 or negative["source_kind"] != positive["source_kind"]:
            raise ValueError("Training rows must be adjacent one-fact pairs")
        code = positive["labels"][0]
        counts[code] += 1
        pairs.extend([(negative, positive, code)] * REPEAT[code])
    if any(count != 48 for count in counts.values()) or len(pairs) != 864:
        raise ValueError("Pair exposure differs from preregistration")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=False, local_files_only=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["Wqkv", "Wo"], modules_to_save=["classifier"]))
    model.to(device)
    prepared = [tuple(tokenizer(make_prompt_v4(row, code), truncation=True, max_length=MAX_LENGTH)
                      for row in (negative, positive)) for negative, positive, code in pairs]

    def collate(batch):
        return tokenizer.pad([item for pair in batch for item in pair], padding=True, return_tensors="pt")

    loader = DataLoader(prepared, batch_size=PAIRS_PER_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), collate_fn=collate)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LR)
    total_steps = EPOCHS * len(loader)
    warmup = max(1, round(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup
        else max(0.0, (total_steps - step) / (total_steps - warmup)))
    output.mkdir(parents=True)
    record = {"status": "development_selection", "amendment": "v9a_pairwise_lora_pretest",
              "source_commit": source_commit, "model": MODEL_ID, "revision": REVISION,
              "base_files_sha256": BASE_HASHES, "train_batch": "batch-0014",
              "development_batch": "batch-0015",
              "train_batch_sha256": sha256(ROOT / "data/synthetic/batch-0014.jsonl"),
              "development_sha256": sha256(ROOT / "data/synthetic/batch-0015.jsonl"),
              "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
              "prompt_sha256": sha256(ROOT / "src/roguard/prompt_v4.py"),
              "study_code_sha256": sha256(ROOT / "src/roguard/mmbert_study.py"),
              "trainer_sha256": sha256(Path(__file__)),
              "protocol_sha256": sha256(ROOT / "docs/V9A_AMENDMENT.md"),
              "seed": SEED, "epochs": EPOCHS, "pairs_per_batch": PAIRS_PER_BATCH,
              "pair_exposures_per_epoch": len(pairs), "unique_pairs": 288,
              "pair_loss_weight": PAIR_LOSS_WEIGHT, "learning_rate": LR,
              "max_length": MAX_LENGTH, "warmup_steps": warmup, "total_steps": total_steps,
              "device": device, "torch_version": torch.__version__,
              "transformers_version": __import__("transformers").__version__,
              "peft_version": __import__("peft").__version__, "candidates": []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for step, batch in enumerate(loader, 1):
            batch = batch.to(device)
            labels = torch.tensor([0, 1] * (batch["input_ids"].shape[0] // 2), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits.float()
            cross_entropy = torch.nn.functional.cross_entropy(logits, labels)
            margins = logits[:, 1] - logits[:, 0]
            pair_loss = torch.nn.functional.softplus(margins[::2] - margins[1::2]).mean()
            loss = cross_entropy + PAIR_LOSS_WEIGHT * pair_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            if step % 36 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} mean_loss={loss_sum / step:.4f}", flush=True)
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        uncalibrated = predict_rows(model, tokenizer, dev_rows, device)
        threshold, report, items, candidate_count = choices_from_scores(dev_rows, uncalibrated)
        dev_path = output / f"epoch-{epoch}-dev.json"
        dev_path.write_text(json.dumps({"batches": ["batch-0015"], "threshold": threshold,
                                        "predictions": items, "report": report},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        candidate = {"epoch": epoch, "mean_train_loss": loss_sum / len(loader),
                     "threshold": threshold, "threshold_candidates": candidate_count,
                     "weight_sha256": sha256(checkpoint / "adapter_model.safetensors"),
                     "dev_predictions_sha256": sha256(dev_path), "report": report}
        record["candidates"].append(candidate)
        print(f"epoch={epoch} dev_pairs={report['exact_pairs']}/48 dev_cards={report['exact_cards']}/96 threshold={threshold:.4f}", flush=True)
    best = max(record["candidates"], key=lambda c: (
        c["report"]["exact_pairs"], c["report"]["exact_cards"],
        -c["report"]["false_review_on_negatives"],
        -abs(c["threshold"] - 0.5), -c["epoch"], -c["threshold"]))
    record["selected_epoch"] = best["epoch"]
    record["selected_threshold"] = best["threshold"]
    record["selected_weight_sha256"] = best["weight_sha256"]
    target = output / "development-selection.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected={best['epoch']} threshold={best['threshold']:.4f} record={target}", flush=True)


if __name__ == "__main__":
    main()
