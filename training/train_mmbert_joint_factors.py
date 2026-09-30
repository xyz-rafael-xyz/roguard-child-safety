"""Train the preregistered V19 joint four-field D1 encoder on abstract pairs."""

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
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS, decision
from roguard.review import load_approved, sha256, taxonomy_sha256
from v18_facts import load_d1_rows_with_facts

ROOT = Path(__file__).resolve().parents[1]
TRAIN_BATCHES = ("batch-0028", "batch-0031", "batch-0032", "batch-0033", "batch-0034")
DEV_BATCH = "batch-0035"
SEED = 20261001
EPOCHS = 3
PAIR_BATCH_SIZE = 4
LR = 5e-5


def load_development() -> list[tuple[dict, dict[str, bool]]]:
    rows = load_approved(ROOT, [DEV_BATCH])
    generator = runpy.run_path(str(ROOT / "training/generate_joint_factor_holdout.py"))
    raw = generator["build_d1_facts"](DEV_BATCH)
    facts = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in raw]
    if (len(rows) != 48 or len(facts) != len(rows) or any(
            row["labels"] != (["D1"] if decision(item) else [])
            for row, item in zip(rows, facts))):
        raise ValueError("V19 development facts differ from attested labels")
    return list(zip(rows, facts))


def score_cards(model, tokenizer, cards: list[tuple[dict, dict]], device: str) -> list[dict]:
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(cards), 16):
            encoded = tokenizer([row["text"] for row, _ in cards[start:start + 16]],
                                padding=True, truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt").to(device)
            scores.extend(torch.sigmoid(model(**encoded).logits.float()).cpu().tolist())
    items = []
    for (row, facts), values in zip(cards, scores):
        fields = dict(zip(FIELDS, map(float, values)))
        predicted = decision({field: fields[field] >= 0.5 for field in FIELDS})
        items.append({"id": row["id"], "expected": row["labels"],
                      "strict_parse": True, "field_scores": fields,
                      "field_expected": facts,
                      "predicted": ["D1"] if predicted else []})
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
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/ro-mmbert-v19")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve the prior V19 training run")
    if (ROOT / "data/synthetic/batch-0036.jsonl").exists():
        raise ValueError("V19 sealed test must remain unopened during development")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit V19 protocol, generator, dev batch and trainer before fitting")
    verify_base(base)
    train = [card for batch in TRAIN_BATCHES for card in load_d1_rows_with_facts(ROOT, batch)]
    dev = load_development()
    if len(train) != 432 or len(dev) != 48:
        raise ValueError("Unexpected V19 training or development card count")
    pairs = []
    for left, right in zip(train[::2], train[1::2]):
        left_row, left_facts = left
        right_row, right_facts = right
        changed = [int(not left_facts[field] and right_facts[field]) for field in FIELDS]
        if (left_row["labels"] or right_row["labels"] != ["D1"] or
                sum(changed) != 1 or any(left_facts[field] and not right_facts[field]
                                         for field in FIELDS)):
            raise ValueError("V19 requires adjacent one-fact negative-to-positive pairs")
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
        labels = torch.tensor([targets for _, _, targets, _ in batch], dtype=torch.float32)
        changed = torch.tensor([mask for _, _, _, mask in batch], dtype=torch.float32)
        return tokenizer.pad(inputs, padding=True, return_tensors="pt"), labels, changed

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
        "status": "development_selection", "study": "v19_joint_factorized_d1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model": MODEL_ID, "revision": REVISION, "base_files_sha256": BASE_HASHES,
        "train_batches": list(TRAIN_BATCHES), "development_batch": DEV_BATCH,
        "train_batch_sha256": {batch: sha256(ROOT / f"data/synthetic/{batch}.jsonl")
                               for batch in TRAIN_BATCHES},
        "development_sha256": sha256(ROOT / f"data/synthetic/{DEV_BATCH}.jsonl"),
        "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
        "generator_sha256": sha256(ROOT / "training/generate_joint_factor_holdout.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(ROOT / "docs/V19_PROTOCOL.md"),
        "seed": SEED, "epochs": EPOCHS, "pair_batch_size": PAIR_BATCH_SIZE,
        "train_pairs": len(pairs), "learning_rate": LR,
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
        for step, (batch, targets, changed) in enumerate(loader, 1):
            batch, targets, changed = batch.to(device), targets.to(device), changed.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits.float().reshape(-1, 2, len(FIELDS))
            binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            margins = ((logits[:, 1] - logits[:, 0]) * changed).sum(dim=-1)
            pair_loss = torch.relu(1.0 - margins).mean()
            invariant = ((torch.sigmoid(logits[:, 1]) - torch.sigmoid(logits[:, 0])).abs()
                         * (1 - changed)).mean()
            loss = binary_loss + 0.25 * pair_loss + 0.05 * invariant
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
        print(f"epoch={epoch} dev_pairs={metrics['exact_pairs']}/24 "
              f"false_reviews={metrics['false_review_on_negatives']}/24 "
              f"field_correct={field_correct}/192", flush=True)
    best = select(record["candidates"])
    record["selected_epoch"] = best["epoch"]
    record["selected_weight_sha256"] = best["weight_sha256"]
    positives = next(item for item in best["report"]["per_category"] if item["category"] == "D1")
    record["development_gate_passed"] = (
        best["report"]["exact_pairs"] >= 21 and
        positives["tp"] >= 21 and
        best["report"]["false_review_on_negatives"] <= 3 and
        best["field_correct"] >= 170)
    target = output / "development-selection.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected={best['epoch']} gate={record['development_gate_passed']} record={target}", flush=True)


if __name__ == "__main__":
    main()
