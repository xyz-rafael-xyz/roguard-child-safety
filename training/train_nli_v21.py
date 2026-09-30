"""Fit V21's Romanian abstract D1 NLI adapter before opening batch 0038."""

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

from roguard.nli_v21 import (BASE_HASHES, FIELDS, HYPOTHESES, MAX_LENGTH,
                             MODEL_ID, REVISION, decision, field_probabilities,
                             verify_base)
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256, taxonomy_sha256
from v18_facts import load_d1_rows_with_facts

ROOT = Path(__file__).resolve().parents[1]
TRAIN_BATCHES = ("batch-0028", "batch-0031", "batch-0032", "batch-0033",
                 "batch-0034", "batch-0036")
DEV_BATCH = "batch-0037"
SEED = 20261003
EPOCHS = 3
PAIR_BATCH_SIZE = 2
LR = 3e-5


def _from_raw(raw: list[dict]) -> list[dict[str, bool]]:
    return [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
             "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
            for item in raw]


def load_cards(batch: str) -> list[tuple[dict, dict[str, bool]]]:
    if batch in TRAIN_BATCHES and batch != "batch-0036":
        return load_d1_rows_with_facts(ROOT, batch)
    if batch == "batch-0036":
        generator = "generate_joint_factor_holdout.py"
    elif batch == DEV_BATCH:
        generator = "generate_balanced_joint_holdout.py"
    else:
        raise ValueError("Unknown V21 source batch")
    rows = [row for row in load_approved(ROOT, [batch]) if row["source_kind"] == "message"]
    module = runpy.run_path(str(ROOT / "training" / generator))
    facts = _from_raw(module["build_d1_facts"](batch))
    if len(rows) != len(facts) or any(
            row["labels"] != (["D1"] if decision(item) else [])
            for row, item in zip(rows, facts)):
        raise ValueError("V21 typed facts differ from attested D1 labels")
    return list(zip(rows, facts))


def score_cards(model, tokenizer, cards: list[tuple[dict, dict]], device: str) -> list[dict]:
    model.eval()
    prompts = [(row["text"], HYPOTHESES[field]) for row, _ in cards for field in FIELDS]
    values = []
    with torch.inference_mode():
        for start in range(0, len(prompts), 16):
            chunk = prompts[start:start + 16]
            encoded = tokenizer([item[0] for item in chunk], [item[1] for item in chunk],
                                padding=True, truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt").to(device)
            values.extend(field_probabilities(model(**encoded).logits))
    if len(values) != len(cards) * len(FIELDS):
        raise ValueError("V21 factor scores are incomplete")
    items = []
    for index, (row, facts) in enumerate(cards):
        scores = dict(zip(FIELDS, values[index * 4:index * 4 + 4]))
        positive = decision({field: scores[field] >= 0.5 for field in FIELDS})
        items.append({"id": row["id"], "expected": row["labels"],
                      "strict_parse": True, "field_scores": scores,
                      "field_expected": facts,
                      "predicted": ["D1"] if positive else []})
    return items


def report(cards: list[tuple[dict, dict]], items: list[dict]) -> tuple[dict, int]:
    pairs = evaluate_pairs([row for row, _ in cards], items)
    correct = sum((item["field_scores"][field] >= 0.5) == facts[field]
                  for (_, facts), item in zip(cards, items) for field in FIELDS)
    return pairs, correct


def select(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda item: (
        item["report"]["exact_pairs"], -item["report"]["false_review_on_negatives"],
        item["field_correct"], -item["epoch"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/ro-nli-v21")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior V21 training run")
    if (ROOT / "data/synthetic/batch-0038.jsonl").exists():
        raise ValueError("V21 sealed test must remain unopened during development")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit V21 protocol, generator, prompts and trainer before fitting")
    verify_base(base)
    train = [item for batch in TRAIN_BATCHES for item in load_cards(batch)]
    dev = load_cards(DEV_BATCH)
    if len(train) != 528 or len(dev) != 96:
        raise ValueError("Unexpected V21 training or development card count")
    positives = [sum(int(facts[field]) for _, facts in train) for field in FIELDS]
    if any(count == 0 or count == len(train) for count in positives):
        raise ValueError("Every V21 field needs both positive and negative training cards")
    class_weights = [((len(train) / (2 * count)),
                      (len(train) / (2 * (len(train) - count)))) for count in positives]
    pairs = []
    for left, right in zip(train[::2], train[1::2]):
        changed = [int(not left[1][field] and right[1][field]) for field in FIELDS]
        if (left[0]["labels"] or right[0]["labels"] != ["D1"] or
                sum(changed) != 1 or any(left[1][field] and not right[1][field]
                                         for field in FIELDS)):
            raise ValueError("V21 requires adjacent negative-to-positive one-fact pairs")
        pairs.append((left, right, changed))
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    foundation = AutoModelForSequenceClassification.from_pretrained(
        base, use_safetensors=True, local_files_only=True).float()
    if foundation.config.id2label != {0: "entailment", 1: "neutral", 2: "contradiction"}:
        raise ValueError("V21 NLI label order differs")
    model = get_peft_model(foundation, LoraConfig(
        task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["query_proj", "value_proj"],
        modules_to_save=["classifier", "pooler"])).to(device)
    encoded = []
    for left, right, changed in pairs:
        inputs = [tokenizer(row["text"], HYPOTHESES[field], truncation=True,
                            max_length=MAX_LENGTH)
                  for row, _ in (left, right) for field in FIELDS]
        targets = [0 if facts[field] else 2 for _, facts in (left, right) for field in FIELDS]
        weights = [class_weights[index][0 if facts[field] else 1]
                   for _, facts in (left, right) for index, field in enumerate(FIELDS)]
        encoded.append((inputs, targets, weights, changed))

    def collate(batch):
        inputs = [item for rows, _, _, _ in batch for item in rows]
        targets = torch.tensor([value for _, labels, _, _ in batch for value in labels], dtype=torch.long)
        weights = torch.tensor([value for _, _, values, _ in batch for value in values], dtype=torch.float32)
        changed = torch.tensor([mask for _, _, _, mask in batch], dtype=torch.float32)
        return tokenizer.pad(inputs, padding=True, return_tensors="pt"), targets, weights, changed

    loader = DataLoader(encoded, batch_size=PAIR_BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), collate_fn=collate)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    total_steps = EPOCHS * len(loader)
    warmup = max(1, round(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup else
        max(0.0, (total_steps - step) / (total_steps - warmup)))
    output.mkdir(parents=True)
    record = {
        "status": "development_selection", "study": "v21_nli_adapted_d1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model": MODEL_ID, "revision": REVISION, "base_files_sha256": BASE_HASHES,
        "train_batches": list(TRAIN_BATCHES), "development_batch": DEV_BATCH,
        "train_batch_sha256": {batch: sha256(ROOT / f"data/synthetic/{batch}.jsonl")
                               for batch in TRAIN_BATCHES},
        "development_sha256": sha256(ROOT / f"data/synthetic/{DEV_BATCH}.jsonl"),
        "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
        "generator_sha256": sha256(ROOT / "training/generate_nli_transfer_holdout.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "prompt_sha256": sha256(ROOT / "src/roguard/nli_v21.py"),
        "protocol_sha256": sha256(ROOT / "docs/V21_PROTOCOL.md"),
        "seed": SEED, "epochs": EPOCHS, "pair_batch_size": PAIR_BATCH_SIZE,
        "train_pairs": len(pairs), "positive_counts": dict(zip(FIELDS, positives)),
        "class_weights": {field: dict(zip(("positive", "negative"), weights))
                          for field, weights in zip(FIELDS, class_weights)},
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
        for step, (batch, targets, weights, changed) in enumerate(loader, 1):
            batch, targets = batch.to(device), targets.to(device)
            weights, changed = weights.to(device), changed.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits.float()
            ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            balanced_ce = (ce * weights).sum() / weights.sum()
            margins = (logits[:, 0] - logits[:, 2]).reshape(-1, 2, len(FIELDS))
            pair_margin = ((margins[:, 1] - margins[:, 0]) * changed).sum(dim=-1)
            pair_loss = torch.relu(1.0 - pair_margin).mean()
            invariant = ((torch.sigmoid(margins[:, 1]) - torch.sigmoid(margins[:, 0])).abs()
                         * (1 - changed)).mean()
            loss = balanced_ce + 0.25 * pair_loss + 0.05 * invariant
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
        dev_path.write_text(json.dumps({"batch": DEV_BATCH, "predictions": items,
                                        "report": metrics, "field_correct": field_correct},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record["candidates"].append({
            "epoch": epoch, "mean_train_loss": loss_total / len(loader),
            "weight_sha256": sha256(checkpoint / "adapter_model.safetensors"),
            "dev_predictions_sha256": sha256(dev_path),
            "report": metrics, "field_correct": field_correct,
        })
        print(f"epoch={epoch} dev_pairs={metrics['exact_pairs']}/48 "
              f"false_reviews={metrics['false_review_on_negatives']}/48 "
              f"field_correct={field_correct}/384", flush=True)
    best = select(record["candidates"])
    record["selected_epoch"] = best["epoch"]
    record["selected_weight_sha256"] = best["weight_sha256"]
    positives_report = next(item for item in best["report"]["per_category"]
                            if item["category"] == "D1")
    record["development_gate_passed"] = (
        best["report"]["exact_pairs"] >= 42 and positives_report["tp"] >= 42 and
        best["report"]["false_review_on_negatives"] <= 6 and
        best["field_correct"] >= 340)
    target = output / "development-selection.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected={best['epoch']} gate={record['development_gate_passed']} record={target}", flush=True)


if __name__ == "__main__":
    main()
