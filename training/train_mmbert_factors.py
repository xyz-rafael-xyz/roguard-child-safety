"""Fit the registered V18 four-factor abstract D1 encoder."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roguard.mmbert_study import BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION, verify_base
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS, decision, make_factor_prompt
from roguard.review import sha256, taxonomy_sha256
from v18_facts import load_d1_rows_with_facts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/ro-mmbert-v16-abstract"
TRAIN_BATCHES = ("batch-0028", "batch-0031", "batch-0032")
DEV_BATCH = "batch-0033"
SEED = 20260930
EPOCHS = 3
BATCH_SIZE = 8
LR = 3e-5


def score_factors(model, tokenizer, cards: list[tuple[dict, dict]], device: str) -> list[dict]:
    model.eval()
    prompts = [make_factor_prompt(row["text"], field)
               for row, _ in cards for field in FIELDS]
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), 16):
            encoded = tokenizer(prompts[start:start + 16], padding=True, truncation=True,
                                max_length=MAX_LENGTH, return_tensors="pt").to(device)
            values = torch.softmax(model(**encoded).logits.float(), dim=-1)[:, 1]
            scores.extend(float(value) for value in values.cpu().tolist())
    result = []
    for index, (row, facts) in enumerate(cards):
        field_scores = dict(zip(FIELDS, scores[index * len(FIELDS):(index + 1) * len(FIELDS)]))
        judgments = {field: value >= 0.5 for field, value in field_scores.items()}
        result.append({"id": row["id"], "expected": row["labels"], "strict_parse": True,
                       "factor_scores": field_scores,
                       "factor_expected": facts,
                       "predicted": ["D1"] if decision(judgments) else []})
    return result


def development_report(cards: list[tuple[dict, dict]], items: list[dict]) -> tuple[dict, int]:
    rows = [row for row, _ in cards]
    report = evaluate_pairs(rows, items)
    factor_correct = sum((item["factor_scores"][field] >= 0.5) == facts[field]
                         for (_, facts), item in zip(cards, items) for field in FIELDS)
    return report, factor_correct


def select(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda entry: (
        entry["report"]["exact_pairs"],
        -entry["report"]["false_review_on_negatives"],
        entry["factor_correct"], -entry["epoch"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/ro-mmbert-v18")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior V18 run")
    verify_base(base)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit V18 protocol, generator, batches and trainer before fitting")
    source_manifest = json.loads((SOURCE / "research.json").read_text(encoding="utf-8"))
    if source_manifest["adapter_weight_sha256"] != sha256(SOURCE / "adapter_model.safetensors"):
        raise ValueError("V16 source weight differs")
    train = [card for batch in TRAIN_BATCHES for card in load_d1_rows_with_facts(ROOT, batch)]
    dev = load_d1_rows_with_facts(ROOT, DEV_BATCH)
    if len(train) != 288 or len(dev) != 48:
        raise ValueError("Unexpected V18 factor card counts")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    foundation = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=False, local_files_only=True)
    model = PeftModel.from_pretrained(foundation, SOURCE, is_trainable=True).to(device)
    tasks = [(make_factor_prompt(row["text"], field), int(facts[field]), field)
             for row, facts in train for field in FIELDS]
    class_counts = {field: {label: sum(task_field == field and target == label
                                      for _, target, task_field in tasks)
                            for label in (0, 1)} for field in FIELDS}
    if any(count == 0 for counts in class_counts.values() for count in counts.values()):
        raise ValueError("Every D1 field needs positive and negative training states")
    encoded = [(tokenizer(prompt, truncation=True, max_length=MAX_LENGTH), label,
                len(train) / (2 * class_counts[field][label]))
               for prompt, label, field in tasks]

    def collate(batch):
        texts, labels, weights = zip(*batch)
        return (tokenizer.pad(list(texts), padding=True, return_tensors="pt"),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(weights, dtype=torch.float32))

    loader = DataLoader(encoded, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), collate_fn=collate)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    total_steps = EPOCHS * len(loader)
    warmup = max(1, round(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup else
        max(0.0, (total_steps - step) / (total_steps - warmup)))
    output.mkdir(parents=True)
    record = {
        "status": "development_selection", "study": "v18_factorized_d1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model": MODEL_ID, "revision": REVISION, "base_files_sha256": BASE_HASHES,
        "source_adapter_weight_sha256": sha256(SOURCE / "adapter_model.safetensors"),
        "source_manifest_sha256": sha256(SOURCE / "research.json"),
        "train_batches": list(TRAIN_BATCHES), "development_batch": DEV_BATCH,
        "train_batch_sha256": {batch: sha256(ROOT / f"data/synthetic/{batch}.jsonl")
                               for batch in TRAIN_BATCHES},
        "development_sha256": sha256(ROOT / f"data/synthetic/{DEV_BATCH}.jsonl"),
        "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
        "prompt_sha256": sha256(ROOT / "src/roguard/prompt_v18.py"),
        "facts_sha256": sha256(ROOT / "training/v18_facts.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(ROOT / "docs/V18_PROTOCOL.md"),
        "prefit_amendment_sha256": sha256(ROOT / "docs/V18_PREFIT_AMENDMENT.md"),
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "factor_task_count": len(tasks), "class_counts": class_counts,
        "learning_rate": LR,
        "max_length": MAX_LENGTH, "warmup_steps": warmup,
        "total_steps": total_steps, "device": device,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "candidates": [],
    }
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_total = 0.0
        for step, (batch, labels, weights) in enumerate(loader, 1):
            batch, labels, weights = batch.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits.float()
            loss = (torch.nn.functional.cross_entropy(logits, labels, reduction="none") * weights).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_total += float(loss.item())
            if step % 48 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} mean_loss={loss_total / step:.4f}", flush=True)
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        items = score_factors(model, tokenizer, dev, device)
        report, factor_correct = development_report(dev, items)
        dev_path = output / f"epoch-{epoch}-dev.json"
        dev_path.write_text(json.dumps({"batch": DEV_BATCH, "predictions": items,
                                        "report": report, "factor_correct": factor_correct},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record["candidates"].append({
            "epoch": epoch, "mean_train_loss": loss_total / len(loader),
            "weight_sha256": sha256(checkpoint / "adapter_model.safetensors"),
            "dev_predictions_sha256": sha256(dev_path),
            "report": report, "factor_correct": factor_correct,
        })
        print(f"epoch={epoch} dev_pairs={report['exact_pairs']}/24 "
              f"false_reviews={report['false_review_on_negatives']}/24 "
              f"factor_correct={factor_correct}/192", flush=True)
    best = select(record["candidates"])
    record["selected_epoch"] = best["epoch"]
    record["selected_weight_sha256"] = best["weight_sha256"]
    d1_metrics = next(entry for entry in best["report"]["per_category"] if entry["category"] == "D1")
    record["development_gate_passed"] = (
        d1_metrics["tp"] >= 21 and best["report"]["false_review_on_negatives"] <= 3)
    target = output / "development-selection.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected={best['epoch']} gate={record['development_gate_passed']} record={target}", flush=True)


if __name__ == "__main__":
    main()
