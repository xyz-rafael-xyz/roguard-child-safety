"""Fit preregistered V20 with balanced four-field and one-fact pair losses."""

from __future__ import annotations

import argparse
import json
import random
import runpy
import subprocess
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roguard.mmbert_study import BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION, verify_base
from roguard.prompt_v18 import FIELDS, decision
from roguard.review import load_approved, sha256, taxonomy_sha256
from train_mmbert_joint_factors import load_development, report, score_cards, select
from v18_facts import load_d1_rows_with_facts

ROOT = Path(__file__).resolve().parents[1]
TRAIN_BATCHES = ("batch-0028", "batch-0031", "batch-0032", "batch-0033",
                 "batch-0034", "batch-0036")
SEED = 20261002
EPOCHS = 5
PAIR_BATCH_SIZE = 4
LR = 8e-5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/ro-mmbert-v20")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior V20 training run")
    if (ROOT / "data/synthetic/batch-0037.jsonl").exists():
        raise ValueError("V20 sealed test must remain unopened during development")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit V20 protocol, generator and trainer before fitting")
    verify_base(base)
    train = [card for batch in TRAIN_BATCHES[:-1]
             for card in load_d1_rows_with_facts(ROOT, batch)]
    extra_rows = load_approved(ROOT, ["batch-0036"])
    prior_generator = runpy.run_path(str(ROOT / "training/generate_joint_factor_holdout.py"))
    extra_raw = prior_generator["build_d1_facts"]("batch-0036")
    extra_facts = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
                    "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
                   for item in extra_raw]
    if len(extra_rows) != len(extra_facts) or any(
            row["labels"] != (["D1"] if decision(facts) else [])
            for row, facts in zip(extra_rows, extra_facts)):
        raise ValueError("V20 additional training facts differ from attested labels")
    train.extend(zip(extra_rows, extra_facts))
    dev = load_development()
    if len(train) != 528 or len(dev) != 48:
        raise ValueError("Unexpected V20 training or development card count")
    positives = [sum(int(facts[field]) for _, facts in train) for field in FIELDS]
    if any(count == 0 or count == len(train) for count in positives):
        raise ValueError("Every V20 field needs both positive and negative training cards")
    weights = [(len(train) - count) / count for count in positives]
    pairs = []
    for left, right in zip(train[::2], train[1::2]):
        changed = [int(not left[1][field] and right[1][field]) for field in FIELDS]
        if (left[0]["labels"] or right[0]["labels"] != ["D1"] or
                sum(changed) != 1 or any(left[1][field] and not right[1][field]
                                         for field in FIELDS)):
            raise ValueError("V20 requires adjacent one-fact negative-to-positive pairs")
        pairs.append((left, right, changed))
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    foundation = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=4, problem_type="multi_label_classification",
        use_safetensors=False, local_files_only=True)
    model = get_peft_model(foundation, LoraConfig(
        task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["Wqkv", "Wo"], modules_to_save=["classifier"])).to(device)
    encoded = [(
        tokenizer(left[0]["text"], truncation=True, max_length=MAX_LENGTH),
        tokenizer(right[0]["text"], truncation=True, max_length=MAX_LENGTH),
        [[float(left[1][field]) for field in FIELDS],
         [float(right[1][field]) for field in FIELDS]], changed,
    ) for left, right, changed in pairs]

    def collate(batch):
        inputs = [item for left, right, _, _ in batch for item in (left, right)]
        targets = torch.tensor([values for _, _, values, _ in batch], dtype=torch.float32)
        changed = torch.tensor([mask for _, _, _, mask in batch], dtype=torch.float32)
        return tokenizer.pad(inputs, padding=True, return_tensors="pt"), targets, changed

    loader = DataLoader(encoded, batch_size=PAIR_BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), collate_fn=collate)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    total_steps = EPOCHS * len(loader)
    warmup = max(1, round(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup else
        max(0.0, (total_steps - step) / (total_steps - warmup)))
    positive_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    output.mkdir(parents=True)
    record = {
        "status": "development_selection", "study": "v20_balanced_joint_d1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model": MODEL_ID, "revision": REVISION, "base_files_sha256": BASE_HASHES,
        "train_batches": list(TRAIN_BATCHES), "development_batch": "batch-0035",
        "train_batch_sha256": {batch: sha256(ROOT / f"data/synthetic/{batch}.jsonl")
                               for batch in TRAIN_BATCHES},
        "development_sha256": sha256(ROOT / "data/synthetic/batch-0035.jsonl"),
        "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
        "generator_sha256": sha256(ROOT / "training/generate_balanced_joint_holdout.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "shared_training_helpers_sha256": sha256(ROOT / "training/train_mmbert_joint_factors.py"),
        "consumed_training_generator_sha256": sha256(ROOT / "training/generate_joint_factor_holdout.py"),
        "protocol_sha256": sha256(ROOT / "docs/V20_PROTOCOL.md"),
        "seed": SEED, "epochs": EPOCHS, "pair_batch_size": PAIR_BATCH_SIZE,
        "train_pairs": len(pairs), "positive_counts": dict(zip(FIELDS, positives)),
        "positive_class_weights": dict(zip(FIELDS, weights)),
        "learning_rate": LR, "max_length": MAX_LENGTH,
        "warmup_steps": warmup, "total_steps": total_steps, "device": device,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "candidates": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_total = 0.0
        for step, (batch, targets, changed) in enumerate(loader, 1):
            batch, targets, changed = batch.to(device), targets.to(device), changed.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits.float().reshape(-1, 2, len(FIELDS))
            binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=positive_weights)
            margins = ((logits[:, 1] - logits[:, 0]) * changed).sum(dim=-1)
            pair_loss = torch.relu(1.0 - margins).mean()
            invariant = ((torch.sigmoid(logits[:, 1]) - torch.sigmoid(logits[:, 0])).abs()
                         * (1 - changed)).mean()
            loss = binary_loss + pair_loss + 0.1 * invariant
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_total += float(loss.item())
            if step % 40 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} mean_loss={loss_total / step:.4f}", flush=True)
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        items = score_cards(model, tokenizer, dev, device)
        metrics, field_correct = report(dev, items)
        dev_path = output / f"epoch-{epoch}-dev.json"
        dev_path.write_text(json.dumps({"batch": "batch-0035", "predictions": items,
                                        "report": metrics, "field_correct": field_correct},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record["candidates"].append({
            "epoch": epoch, "mean_train_loss": loss_total / len(loader),
            "weight_sha256": sha256(checkpoint / "adapter_model.safetensors"),
            "dev_predictions_sha256": sha256(dev_path),
            "report": metrics, "field_correct": field_correct,
        })
        print(f"epoch={epoch} dev_pairs={metrics['exact_pairs']}/24 "
              f"false_reviews={metrics['false_review_on_negatives']}/24 "
              f"field_correct={field_correct}/192", flush=True)
    best = select(record["candidates"])
    record["selected_epoch"] = best["epoch"]
    record["selected_weight_sha256"] = best["weight_sha256"]
    positives_metrics = next(item for item in best["report"]["per_category"]
                             if item["category"] == "D1")
    record["development_gate_passed"] = (
        best["report"]["exact_pairs"] >= 21 and positives_metrics["tp"] >= 21 and
        best["report"]["false_review_on_negatives"] <= 3 and
        best["field_correct"] >= 170)
    target = output / "development-selection.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected={best['epoch']} gate={record['development_gate_passed']} record={target}", flush=True)


if __name__ == "__main__":
    main()
