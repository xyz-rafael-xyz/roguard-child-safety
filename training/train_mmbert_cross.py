"""Fit v10 pairwise LoRA with negative tasks from the other response category."""

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
from roguard.prompt_v4 import make_prompt_v4
from roguard.review import CATEGORIES, load_train_dev, sha256, taxonomy_sha256
from train_mmbert_pairs import choices_from_scores

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260930
EPOCHS = 3
PAIRS_PER_BATCH = 2
LR = 1e-4
PAIR_WEIGHT = 0.5
CROSS_WEIGHT = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/ro-mmbert-v10")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("V10 output already exists; preserve prior run")
    verify_base(base)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit study code, protocol, and sealed test before fitting")
    train_rows, dev_rows = load_train_dev(ROOT, ["batch-0014", "batch-0015"], "ro")
    if len(train_rows) != 576 or len(dev_rows) != 96:
        raise ValueError("Unexpected fit/development cardinality")
    pairs = []
    counts = {code: 0 for code in CATEGORIES}
    for negative, positive in zip(train_rows[::2], train_rows[1::2]):
        if (negative["labels"] or len(positive["labels"]) != 1 or
                negative["source_kind"] != positive["source_kind"]):
            raise ValueError("Fit data must contain adjacent one-fact pairs")
        code = positive["labels"][0]
        counts[code] += 1
        pairs.extend([(negative, positive, code)] * 3)
    response_pairs = sum(code in ("A1", "S1") for _, _, code in pairs)
    if (len(pairs) != 864 or response_pairs != 288 or
            any(count != 48 for count in counts.values())):
        raise ValueError("Primary or cross-category exposure differs")
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
    prepared = []
    for negative, positive, code in pairs:
        prompts = [make_prompt_v4(row, code) for row in (negative, positive)]
        if code in ("A1", "S1"):
            other = "S1" if code == "A1" else "A1"
            prompts.extend(make_prompt_v4(row, other) for row in (negative, positive))
        prepared.append(tuple(tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)
                              for prompt in prompts))

    def collate(batch):
        texts, primary, auxiliary = [], [], []
        for pair in batch:
            offset = len(texts)
            texts.extend(pair)
            primary.extend((offset, offset + 1))
            if len(pair) == 4:
                auxiliary.extend((offset + 2, offset + 3))
        return tokenizer.pad(texts, padding=True, return_tensors="pt"), primary, auxiliary

    loader = DataLoader(prepared, batch_size=PAIRS_PER_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), collate_fn=collate)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LR)
    total_steps = EPOCHS * len(loader)
    warmup = max(1, round(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup
        else max(0.0, (total_steps - step) / (total_steps - warmup)))
    output.mkdir(parents=True)
    record = {"status": "development_selection", "study": "v10_cross_category_negative",
              "source_commit": source_commit, "model": MODEL_ID, "revision": REVISION,
              "base_files_sha256": BASE_HASHES, "train_batch": "batch-0014",
              "development_batch": "batch-0015",
              "train_batch_sha256": sha256(ROOT / "data/synthetic/batch-0014.jsonl"),
              "development_sha256": sha256(ROOT / "data/synthetic/batch-0015.jsonl"),
              "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
              "prompt_sha256": sha256(ROOT / "src/roguard/prompt_v4.py"),
              "study_code_sha256": sha256(ROOT / "src/roguard/mmbert_study.py"),
              "trainer_sha256": sha256(Path(__file__)),
              "selector_sha256": sha256(ROOT / "training/train_mmbert_pairs.py"),
              "protocol_sha256": sha256(ROOT / "docs/V10_PROTOCOL.md"),
              "seed": SEED, "epochs": EPOCHS, "pairs_per_batch": PAIRS_PER_BATCH,
              "primary_pair_exposures_per_epoch": len(pairs),
              "cross_negative_task_exposures_per_epoch": 2 * response_pairs,
              "unique_pairs": 288, "pair_loss_weight": PAIR_WEIGHT,
              "cross_loss_weight": CROSS_WEIGHT, "learning_rate": LR,
              "max_length": MAX_LENGTH, "warmup_steps": warmup, "total_steps": total_steps,
              "device": device, "torch_version": torch.__version__,
              "transformers_version": __import__("transformers").__version__,
              "peft_version": __import__("peft").__version__, "candidates": []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for step, (batch, primary, auxiliary) in enumerate(loader, 1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits.float()
            main_logits = logits[primary]
            labels = torch.tensor([0, 1] * (len(primary) // 2), device=device)
            primary_loss = torch.nn.functional.cross_entropy(main_logits, labels)
            margins = main_logits[:, 1] - main_logits[:, 0]
            pair_loss = torch.nn.functional.softplus(margins[::2] - margins[1::2]).mean()
            loss = primary_loss + PAIR_WEIGHT * pair_loss
            if auxiliary:
                cross_logits = logits[auxiliary]
                cross_targets = torch.zeros(len(auxiliary), dtype=torch.long, device=device)
                loss = loss + CROSS_WEIGHT * torch.nn.functional.cross_entropy(cross_logits, cross_targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            if step % 72 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} mean_loss={loss_sum / step:.4f}", flush=True)
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        uncalibrated = predict_rows(model, tokenizer, dev_rows, device)
        threshold, report, items, candidate_count = choices_from_scores(dev_rows, uncalibrated)
        dev_path = output / f"epoch-{epoch}-dev.json"
        dev_path.write_text(json.dumps({"batches": ["batch-0015"], "threshold": threshold,
                                        "predictions": items, "report": report},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record["candidates"].append({
            "epoch": epoch, "mean_train_loss": loss_sum / len(loader),
            "threshold": threshold, "threshold_candidates": candidate_count,
            "weight_sha256": sha256(checkpoint / "adapter_model.safetensors"),
            "dev_predictions_sha256": sha256(dev_path), "report": report})
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
