"""Continue frozen V16 on balanced consumed D1 surfaces; select worst-surface dev."""

from __future__ import annotations

import argparse
import json
import random
import runpy
import subprocess
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roguard.mmbert_study import (BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION,
                                  predict_rows, verify_adapter_head, verify_base)
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import make_prompt_v4
from roguard.review import load_approved, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/ro-mmbert-v16-abstract"
SOURCE_WEIGHT_SHA256 = "9f3ccab9b75b9aad5b42739518348bc3af84fd8037d17f3b49ecdb741a858f1d"
TRAIN_BATCHES = ("batch-0028", "batch-0031", "batch-0032", "batch-0033",
                 "batch-0034", "batch-0035", "batch-0036")
DEV_BATCHES = ("batch-0037", "batch-0038")
EXPECTED_PAIRS = (96, 24, 24, 24, 48, 24, 48)
SEED = 20261004
EPOCHS = 3
SAMPLE_PAIRS_PER_GROUP = 24
PAIRS_PER_BATCH = 2
LR = 2e-5
PAIR_WEIGHT = 0.5


def d1_rows(batch: str) -> list[dict]:
    rows = [row for row in load_approved(ROOT, [batch])
            if row["language"] == "ro" and row["source_kind"] == "message"]
    if len(rows) % 2 or any(left["labels"] or right["labels"] != ["D1"]
                              for left, right in zip(rows[::2], rows[1::2])):
        raise ValueError(f"Expected adjacent Romanian D1 pairs: {batch}")
    return rows


def development_reports(rows_by_batch: dict[str, list[dict]],
                        items_by_batch: dict[str, list[dict]]) -> dict:
    reports = {}
    for batch in DEV_BATCHES:
        rows, items = rows_by_batch[batch], items_by_batch[batch]
        reports[batch] = evaluate_pairs(rows, items)
        for half in range(2):
            start = half * 48
            reports[f"{batch}-surface-{half + 1}"] = evaluate_pairs(
                rows[start:start + 48], items[start:start + 48])
    return reports


def selection_summary(reports: dict) -> dict:
    surface_pairs = [reports[f"{batch}-surface-{half}"]["exact_pairs"]
                     for batch in DEV_BATCHES for half in (1, 2)]
    total_pairs = sum(reports[batch]["exact_pairs"] for batch in DEV_BATCHES)
    false_reviews = sum(reports[batch]["false_review_on_negatives"]
                        for batch in DEV_BATCHES)
    positive_recovered = sum(next(item for item in reports[batch]["per_category"]
                                  if item["category"] == "D1")["tp"]
                             for batch in DEV_BATCHES)
    return {"surface_pairs": surface_pairs, "worst_surface_pairs": min(surface_pairs),
            "total_pairs": total_pairs, "false_reviews": false_reviews,
            "positive_recovered": positive_recovered}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/ro-mmbert-v22")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("V22 output already exists; preserve prior run")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit the V22 protocol and trainer before fitting")
    verify_base(base)
    source = json.loads((SOURCE / "research.json").read_text(encoding="utf-8"))
    if (source.get("adapter_weight_sha256") != SOURCE_WEIGHT_SHA256 or
            source.get("shared_threshold") != 0.5 or
            sha256(SOURCE / "adapter_model.safetensors") != SOURCE_WEIGHT_SHA256 or
            source.get("prompt_sha256") != sha256(ROOT / "src/roguard/prompt_v4.py")):
        raise ValueError("Frozen V16 source differs from V22 registration")
    verify_adapter_head(SOURCE)
    rows_by_batch = {batch: d1_rows(batch) for batch in TRAIN_BATCHES + DEV_BATCHES}
    if ([len(rows_by_batch[batch]) // 2 for batch in TRAIN_BATCHES] != list(EXPECTED_PAIRS) or
            any(len(rows_by_batch[batch]) != 96 for batch in DEV_BATCHES)):
        raise ValueError("V22 batch cardinality differs from registration")
    # The unchanged V16 comparator has already been frozen on both consumed dev batches.
    for script in ("verify_mmbert_v20.py", "verify_nli_v21.py"):
        runpy.run_path(str(ROOT / "eval" / script))["verify"](ROOT)
    baselines = {}
    for batch, file in (("batch-0037", "ro-mmbert-v20-test-0037.json"),
                        ("batch-0038", "ro-nli-v21-test-0038.json")):
        saved = json.loads((ROOT / "eval/runs" / file).read_text(encoding="utf-8"))
        baselines[batch] = saved["predictions"]["v16"]
    baseline = selection_summary(development_reports(rows_by_batch, baselines))
    if (baseline["surface_pairs"] != [10, 23, 18, 24] or
            baseline["total_pairs"] != 75 or baseline["false_reviews"] != 13 or
            baseline["positive_recovered"] != 85):
        raise ValueError("Frozen V16 development comparator differs")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    foundation = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=False, local_files_only=True)
    model = PeftModel.from_pretrained(foundation, SOURCE, is_trainable=True).to(device)
    prepared = {}
    for batch in TRAIN_BATCHES:
        rows = rows_by_batch[batch]
        prepared[batch] = [tuple(tokenizer(make_prompt_v4(row, "D1"),
                                           truncation=True, max_length=MAX_LENGTH)
                                 for row in pair)
                           for pair in zip(rows[::2], rows[1::2])]

    def collate(pairs):
        return tokenizer.pad([item for pair in pairs for item in pair],
                             padding=True, return_tensors="pt")

    steps_per_epoch = len(TRAIN_BATCHES) * SAMPLE_PAIRS_PER_GROUP // PAIRS_PER_BATCH
    total_steps = EPOCHS * steps_per_epoch
    warmup = max(1, round(total_steps * 0.1))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup
        else max(0.0, (total_steps - step) / (total_steps - warmup)))
    output.mkdir(parents=True)
    record = {
        "status": "development_selection_only", "study": "v22_balanced_worst_surface_d1",
        "source_commit": source_commit, "model": MODEL_ID, "revision": REVISION,
        "base_files_sha256": BASE_HASHES,
        "source_adapter_weight_sha256": SOURCE_WEIGHT_SHA256,
        "source_manifest_sha256": sha256(SOURCE / "research.json"),
        "train_batches": list(TRAIN_BATCHES), "development_batches": list(DEV_BATCHES),
        "train_batch_sha256": {batch: sha256(ROOT / f"data/synthetic/{batch}.jsonl")
                               for batch in TRAIN_BATCHES},
        "development_sha256": {batch: sha256(ROOT / f"data/synthetic/{batch}.jsonl")
                                for batch in DEV_BATCHES},
        "baseline_run_sha256": {
            "batch-0037": sha256(ROOT / "eval/runs/ro-mmbert-v20-test-0037.json"),
            "batch-0038": sha256(ROOT / "eval/runs/ro-nli-v21-test-0038.json")},
        "baseline_summary": baseline,
        "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
        "prompt_sha256": sha256(ROOT / "src/roguard/prompt_v4.py"),
        "study_code_sha256": sha256(ROOT / "src/roguard/mmbert_study.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(ROOT / "docs/V22_PROTOCOL.md"),
        "seed": SEED, "epochs": EPOCHS, "pairs_per_group_per_epoch": SAMPLE_PAIRS_PER_GROUP,
        "pairs_per_batch": PAIRS_PER_BATCH, "learning_rate": LR,
        "pair_loss_weight": PAIR_WEIGHT, "max_length": MAX_LENGTH,
        "warmup_steps": warmup, "total_steps": total_steps,
        "device": device, "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__, "candidates": [],
    }
    for epoch in range(1, EPOCHS + 1):
        rng = random.Random(SEED + epoch)
        samples = []
        for batch in TRAIN_BATCHES:
            samples.extend(rng.sample(prepared[batch], SAMPLE_PAIRS_PER_GROUP))
        rng.shuffle(samples)
        loader = DataLoader(samples, batch_size=PAIRS_PER_BATCH, shuffle=False, collate_fn=collate)
        model.train()
        loss_sum = 0.0
        for step, encoded in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded.to(device)).logits.float()
            labels = torch.tensor([0, 1] * (logits.shape[0] // 2), device=device)
            primary = torch.nn.functional.cross_entropy(logits, labels)
            margins = logits[:, 1] - logits[:, 0]
            pair_loss = torch.nn.functional.softplus(margins[::2] - margins[1::2]).mean()
            loss = primary + PAIR_WEIGHT * pair_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            if step % 28 == 0:
                print(f"epoch={epoch} step={step}/{steps_per_epoch} mean_loss={loss_sum / step:.4f}", flush=True)
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        items_by_batch = {batch: predict_rows(model, tokenizer, rows_by_batch[batch],
                                              device, threshold=0.5)
                          for batch in DEV_BATCHES}
        reports = development_reports(rows_by_batch, items_by_batch)
        summary = selection_summary(reports)
        dev_path = output / f"epoch-{epoch}-dev.json"
        dev_path.write_text(json.dumps({"batches": list(DEV_BATCHES),
                                        "predictions": items_by_batch,
                                        "reports": reports, "summary": summary},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record["candidates"].append({
            "epoch": epoch, "mean_train_loss": loss_sum / steps_per_epoch,
            "weight_sha256": sha256(checkpoint / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256(checkpoint / "adapter_config.json"),
            "dev_predictions_sha256": sha256(dev_path), "summary": summary,
        })
        print(f"epoch={epoch} dev_surface_pairs={summary['surface_pairs']} total={summary['total_pairs']}/96 false_reviews={summary['false_reviews']}/96", flush=True)
    best = max(record["candidates"], key=lambda item: (
        item["summary"]["worst_surface_pairs"], item["summary"]["total_pairs"],
        -item["summary"]["false_reviews"], -item["epoch"]))
    summary = best["summary"]
    record.update(selected_epoch=best["epoch"], selected_weight_sha256=best["weight_sha256"],
                  development_gate_passed=(summary["worst_surface_pairs"] >= 18 and
                                           summary["total_pairs"] >= 84 and
                                           summary["positive_recovered"] >= 90 and
                                           summary["false_reviews"] <= 8 and
                                           summary["total_pairs"] > baseline["total_pairs"]))
    target = output / "development-selection.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected_epoch={best['epoch']} gate={record['development_gate_passed']} record={target}", flush=True)


if __name__ == "__main__":
    main()
