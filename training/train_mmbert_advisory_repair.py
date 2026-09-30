"""Continue frozen v11 mmBERT on abstract Romanian D1/S1 prose pairs."""

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

from roguard.mmbert_study import (BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION,
                                  predict_rows, verify_base)
from roguard.prompt_v4 import make_prompt_v4
from roguard.review import load_train_dev, sha256, taxonomy_sha256
from train_mmbert_pairs import choices_from_scores

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/ro-mmbert-v11-abstract"
SEED = 20260930
EPOCHS = 3
PAIRS_PER_BATCH = 2
LR = 3e-5
PAIR_WEIGHT = 0.5
CROSS_WEIGHT = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/ro-mmbert-v16")
    args = parser.parse_args()
    base, output = args.base_model_path.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("V16 output already exists; preserve prior run")
    verify_base(base)
    source_manifest = json.loads((SOURCE / "research.json").read_text(encoding="utf-8"))
    if (source_manifest.get("adapter_weight_sha256") !=
            sha256(SOURCE / "adapter_model.safetensors")):
        raise ValueError("Frozen v11 source adapter differs")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit protocol, generator, batches, and trainer before fitting")
    train_rows, dev_rows = load_train_dev(
        ROOT, ["batch-0028", "batch-0029"], "ro", required_categories=("D1", "S1"))
    if len(train_rows) != 384 or len(dev_rows) != 48:
        raise ValueError("Unexpected V16 train/development cardinality")
    pairs = []
    counts = {"D1": 0, "S1": 0}
    for negative, positive in zip(train_rows[::2], train_rows[1::2]):
        if (negative["labels"] or len(positive["labels"]) != 1 or
                positive["labels"][0] not in counts or
                negative["source_kind"] != positive["source_kind"]):
            raise ValueError("V16 fit data must contain D1/S1 one-fact pairs")
        code = positive["labels"][0]
        counts[code] += 1
        pairs.append((negative, positive, code))
    if len(pairs) != 192 or counts != {"D1": 96, "S1": 96}:
        raise ValueError("V16 exposure differs from registration")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    foundation = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=False, local_files_only=True)
    model = PeftModel.from_pretrained(foundation, SOURCE, is_trainable=True)
    model.to(device)
    prepared = []
    for negative, positive, code in pairs:
        prompts = [make_prompt_v4(row, code) for row in (negative, positive)]
        if code == "S1":
            prompts.extend(make_prompt_v4(row, "A1") for row in (negative, positive))
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
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad), lr=LR)
    total_steps = EPOCHS * len(loader)
    warmup = max(1, round(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup
        else max(0.0, (total_steps - step) / (total_steps - warmup)))
    output.mkdir(parents=True)
    record = {"status": "development_selection", "study": "v16_advisory_prose_repair",
              "source_commit": source_commit, "model": MODEL_ID, "revision": REVISION,
              "base_files_sha256": BASE_HASHES,
              "source_adapter_weight_sha256": source_manifest["adapter_weight_sha256"],
              "source_manifest_sha256": sha256(SOURCE / "research.json"),
              "train_batch": "batch-0028", "development_batch": "batch-0029",
              "train_batch_sha256": sha256(ROOT / "data/synthetic/batch-0028.jsonl"),
              "development_sha256": sha256(ROOT / "data/synthetic/batch-0029.jsonl"),
              "taxonomy_sha256": taxonomy_sha256(ROOT, "ro"),
              "prompt_sha256": sha256(ROOT / "src/roguard/prompt_v4.py"),
              "study_code_sha256": sha256(ROOT / "src/roguard/mmbert_study.py"),
              "trainer_sha256": sha256(Path(__file__)),
              "selector_sha256": sha256(ROOT / "training/train_mmbert_pairs.py"),
              "protocol_sha256": sha256(ROOT / "docs/V16_PROTOCOL.md"),
              "prefit_amendment_sha256": sha256(ROOT / "docs/V16_PREFIT_AMENDMENT.md"),
              "seed": SEED, "epochs": EPOCHS, "pairs_per_batch": PAIRS_PER_BATCH,
              "primary_pair_exposures_per_epoch": len(pairs),
              "cross_negative_task_exposures_per_epoch": 192,
              "pair_loss_weight": PAIR_WEIGHT,
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
            if step % 48 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} mean_loss={loss_sum / step:.4f}", flush=True)
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        uncalibrated = predict_rows(model, tokenizer, dev_rows, device)
        threshold, report, items, candidate_count = choices_from_scores(dev_rows, uncalibrated)
        dev_path = output / f"epoch-{epoch}-dev.json"
        dev_path.write_text(json.dumps({"batches": ["batch-0029"], "threshold": threshold,
                                        "predictions": items, "report": report},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record["candidates"].append({
            "epoch": epoch, "mean_train_loss": loss_sum / len(loader),
            "threshold": threshold, "threshold_candidates": candidate_count,
            "weight_sha256": sha256(checkpoint / "adapter_model.safetensors"),
            "dev_predictions_sha256": sha256(dev_path), "report": report})
        print(f"epoch={epoch} dev_pairs={report['exact_pairs']}/24 dev_cards={report['exact_cards']}/48 threshold={threshold:.4f}", flush=True)
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
