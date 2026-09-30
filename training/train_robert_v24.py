"""Fit a pinned Romanian BERT D1 adapter on consumed train/dev abstract cards."""

from __future__ import annotations

import argparse
import json
import math
import random
import runpy
import subprocess
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roguard.mmbert_study import MAX_LENGTH, predict_rows
from roguard.prompt_v4 import make_prompt_v4
from roguard.review import sha256, taxonomy_sha256
from train_mmbert_v23 import (DEV_BATCHES, TRAIN_BATCHES, d1_rows,
                              development_reports, selection_summary)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dumitrescustefan/bert-base-romanian-cased-v1"
REVISION = "37fb0ffb4bc4f7c4cde429626775685fb18f234f"
BASE_HASHES = {
    "config.json": "53d69f53eb2fa0a9e95ae77508ad527dc1ce658a74968a565fbae5cc29495542",
    "model.safetensors": "c2471091c9b2613f671fc46f71023b10d8d3dee5e1b1fe2c6c4d3fce1b4b8508",
    "vocab.txt": "1c8630d8abddcb7d36a51f32a25b72084dd12439bb223ee25cca7e81643acef9",
    "tokenizer_config.json": "adac4eb1158c23ecf85b5215105c1d0bcc42e34d6a9f82d637018f4b7f0b1aa2",
}
EXPECTED_TRAIN_PAIRS = (96, 24, 24, 24, 48, 24, 48)
SEED = 20261006
EPOCHS = 6
SAMPLE_PAIRS_PER_GROUP = 24
PAIRS_PER_BATCH = 2
LR_LORA = 2e-4
LR_CLASSIFIER = 1e-3
PAIR_WEIGHT = 0.5


def verify_base(path: Path) -> None:
    for name, digest in BASE_HASHES.items():
        if sha256(path / name) != digest:
            raise ValueError(f"Romanian BERT base hash differs: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/ro-bert-v24")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("V24 output already exists; preserve previous runs")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit the V24 protocol and trainer before fitting")
    verify_base(base)
    runpy.run_path(str(ROOT / "eval/verify_committed_v23_selection.py"))["verify"](ROOT)
    rows_by_batch = {batch: d1_rows(batch) for batch in TRAIN_BATCHES + DEV_BATCHES}
    if ([len(rows_by_batch[batch]) // 2 for batch in TRAIN_BATCHES] !=
            list(EXPECTED_TRAIN_PAIRS) or
            any(len(rows_by_batch[batch]) != 96 for batch in DEV_BATCHES)):
        raise ValueError("V24 train/development cardinality differs")
    baseline_predictions = {}
    for batch, filename in (("batch-0037", "ro-mmbert-v20-test-0037.json"),
                            ("batch-0038", "ro-nli-v21-test-0038.json")):
        baseline_predictions[batch] = json.loads((ROOT / "eval/runs" / filename).read_text(
            encoding="utf-8"))["predictions"]["v16"]
    baseline = selection_summary(development_reports(rows_by_batch, baseline_predictions))
    if baseline != {"surface_pairs": [10, 23, 18, 24], "worst_surface_pairs": 10,
                    "total_pairs": 75, "false_reviews": 13, "positive_recovered": 85}:
        raise ValueError("Frozen V16 comparator differs")
    v23 = json.loads((ROOT / "eval/runs/ro-mmbert-v23-dev-selection.json").read_text(
        encoding="utf-8"))["candidates"]
    v23_best = max(v23, key=lambda item: (
        item["summary"]["worst_surface_pairs"], item["summary"]["total_pairs"],
        -item["summary"]["false_reviews"], -item["epoch"]))["summary"]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    lengths = [len(tokenizer(make_prompt_v4(row, "D1"), add_special_tokens=True)["input_ids"])
               for batch in TRAIN_BATCHES + DEV_BATCHES for row in rows_by_batch[batch]]
    if max(lengths) > MAX_LENGTH:
        raise ValueError("V24 prompt would be truncated")
    foundation = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=True, local_files_only=True)
    model = get_peft_model(foundation, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["query", "value"], modules_to_save=["classifier"])).to(device)
    prepared = {}
    for batch in TRAIN_BATCHES:
        rows = rows_by_batch[batch]
        prepared[batch] = [tuple(tokenizer(make_prompt_v4(row, "D1")) for row in pair)
                           for pair in zip(rows[::2], rows[1::2])]

    def collate(grouped_pairs):
        return tokenizer.pad([item for _, pair in grouped_pairs for item in pair],
                             padding=True, return_tensors="pt")

    classifier = [parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad and "classifier" in name]
    lora = [parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name]
    if not classifier or not lora or sum(p.numel() for p in classifier + lora) != sum(
            p.numel() for p in model.parameters() if p.requires_grad):
        raise ValueError("V24 trainable parameter groups differ")
    optimizer = torch.optim.AdamW([
        {"params": lora, "lr": LR_LORA},
        {"params": classifier, "lr": LR_CLASSIFIER},
    ])
    steps_per_epoch = len(TRAIN_BATCHES) * SAMPLE_PAIRS_PER_GROUP // PAIRS_PER_BATCH
    total_steps = EPOCHS * steps_per_epoch
    warmup = max(1, round(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup
        else max(0.0, (total_steps - step) / (total_steps - warmup)))
    output.mkdir(parents=True)
    record = {
        "status": "development_selection_only", "study": "v24_native_romanian_bert_d1",
        "source_commit": source_commit, "model": MODEL_ID, "revision": REVISION,
        "base_files_sha256": BASE_HASHES, "train_batches": list(TRAIN_BATCHES),
        "development_batches": list(DEV_BATCHES),
        "train_batch_sha256": {batch: sha256(ROOT / f"data/synthetic/{batch}.jsonl")
                               for batch in TRAIN_BATCHES},
        "development_sha256": {batch: sha256(ROOT / f"data/synthetic/{batch}.jsonl")
                               for batch in DEV_BATCHES},
        "v16_baseline_summary": baseline, "v23_development_summary": v23_best,
        "v23_selection_sha256": sha256(ROOT / "eval/runs/ro-mmbert-v23-dev-selection.json"),
        "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
        "prompt_sha256": sha256(ROOT / "src/roguard/prompt_v4.py"),
        "predictor_sha256": sha256(ROOT / "src/roguard/mmbert_study.py"),
        "data_selector_sha256": sha256(ROOT / "training/train_mmbert_v23.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(ROOT / "docs/V24_PROTOCOL.md"),
        "seed": SEED, "epochs": EPOCHS,
        "pairs_per_group_per_epoch": SAMPLE_PAIRS_PER_GROUP,
        "pairs_per_batch": PAIRS_PER_BATCH, "learning_rate_lora": LR_LORA,
        "learning_rate_classifier": LR_CLASSIFIER, "pair_loss_weight": PAIR_WEIGHT,
        "max_length": MAX_LENGTH, "max_prompt_tokens": max(lengths),
        "warmup_steps": warmup, "total_steps": total_steps,
        "trainable_parameters": sum(p.numel() for p in classifier + lora),
        "device": device, "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__, "candidates": [],
    }
    for epoch in range(1, EPOCHS + 1):
        rng = random.Random(SEED + epoch)
        samples = []
        for batch in TRAIN_BATCHES:
            samples.extend((batch, pair) for pair in
                           rng.sample(prepared[batch], SAMPLE_PAIRS_PER_GROUP))
        rng.shuffle(samples)
        loader = DataLoader(samples, batch_size=PAIRS_PER_BATCH, shuffle=False,
                            collate_fn=collate)
        model.train()
        loss_sum = 0.0
        for step, encoded in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded.to(device)).logits.float()
            labels = torch.tensor([0, 1] * (logits.shape[0] // 2), device=device)
            primary = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none").reshape(-1, 2).mean(dim=1)
            margins = logits[:, 1] - logits[:, 0]
            ranking = torch.nn.functional.softplus(margins[::2] - margins[1::2])
            loss = (primary + PAIR_WEIGHT * ranking).mean()
            if not math.isfinite(float(loss.detach().cpu())):
                raise ValueError("Nonfinite V24 training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            if step % 28 == 0:
                print(f"epoch={epoch} step={step}/{steps_per_epoch} mean_loss={loss_sum / step:.4f}",
                      flush=True)
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
