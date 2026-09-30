"""Fit one exploratory bilingual D1/S1 encoder before opening the V25 test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval.v25_metrics import CELLS, evaluate, select_thresholds
from roguard.mmbert_study import BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION, verify_base
from roguard.review import sha256, taxonomy_sha256
from roguard.v25_abstract import make_input, score_rows
from training.generate_v25_synthetic import (COMMITMENT, DATA, PUBLIC_SEEDS, ROOT,
                                             build_rows)

SEED = 20261001
EPOCHS = 4
PAIRS_PER_BATCH = 4
LR = 3e-5
PAIR_WEIGHT = 0.5
ARTIFACT = ROOT / "models/bi-mmbert-v25-abstract"


def verify_inputs(root: Path) -> tuple[dict, list[dict], list[dict], str]:
    root = root.resolve()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Commit the V25 protocol, generator, data, and trainer before fitting")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                     text=True).strip()
    record = json.loads((root / COMMITMENT).read_text(encoding="utf-8"))
    if (record.get("status") != "sealed_same_origin_synthetic_test_not_yet_generated" or
            record.get("test_rows_at_registration") != 0 or
            record.get("generator_sha256") != sha256(root / "training/generate_v25_synthetic.py") or
            record.get("romanian_wording_sha256") != sha256(root / "training/v25_ro.py") or
            record.get("ukrainian_wording_sha256") != sha256(root / "training/v25_uk.py") or
            record.get("taxonomy_sha256") != {
                language: taxonomy_sha256(root, language) for language in ("ro", "uk")
            } or (root / DATA / "test.jsonl").exists()):
        raise ValueError("V25 pretest registration or taxonomy differs")
    rows = {}
    for split in PUBLIC_SEEDS:
        path = root / DATA / f"{split}.jsonl"
        if sha256(path) != record[f"{split}_sha256"]:
            raise ValueError(f"V25 {split} bytes differ from pretest commitment")
        actual = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if actual != build_rows(split, PUBLIC_SEEDS[split]):
            raise ValueError(f"V25 {split} does not reproduce its frozen generator")
        rows[split] = actual
    return record, rows["train"], rows["dev"], commit


def char_baseline_model(train_rows: list[dict]):
    """Fit the frozen lexical comparator on training rows alone."""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                 min_df=2, max_features=60000)
    inputs = [make_input(row) for row in train_rows]
    features = vectorizer.fit_transform(inputs)
    labels = [int(bool(row["labels"])) for row in train_rows]
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    classifier.fit(features, labels)
    return vectorizer, classifier


def char_baseline_scores(model, rows: list[dict]) -> list[float]:
    vectorizer, classifier = model
    return classifier.predict_proba(vectorizer.transform(
        [make_input(row) for row in rows]))[:, 1].tolist()


def fit_char_baseline(train_rows: list[dict], dev_rows: list[dict]) -> tuple[dict, list[float]]:
    """A fixed, inexpensive lexical comparator; never fit on test rows."""
    model = char_baseline_model(train_rows)
    scores = char_baseline_scores(model, dev_rows)
    thresholds = select_thresholds(dev_rows, scores)
    return {"model": "tfidf_char_3_to_5_logistic_regression",
            "thresholds": thresholds, "report": evaluate(dev_rows, scores, thresholds),
            "features": len(model[0].vocabulary_), "scores": scores}, scores


def _selection_key(candidate: dict) -> tuple:
    cells = candidate["report"]["cells"]
    return (min(cells[cell]["exact_pairs"] for cell in CELLS),
            sum(cells[cell]["exact_pairs"] for cell in CELLS),
            sum(cells[cell]["exact_cards"] for cell in CELLS),
            -sum(cells[cell]["false_reviews"] for cell in CELLS),
            -candidate["epoch"])


def train(base: Path, output: Path) -> dict:
    record, train_rows, dev_rows, source_commit = verify_inputs(ROOT)
    base, output = base.resolve(), output.resolve()
    if output.exists() or ARTIFACT.exists():
        raise FileExistsError("Preserve any existing V25 fitting output or selected artifact")
    verify_base(base)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    texts = [make_input(row) for row in train_rows + dev_rows]
    lengths = [len(tokenizer(text, add_special_tokens=True)["input_ids"]) for text in texts]
    if max(lengths) > MAX_LENGTH:
        raise ValueError("V25 prompt exceeds its frozen token cap")
    foundation = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=False, local_files_only=True)
    model = get_peft_model(foundation, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["Wqkv", "Wo"], modules_to_save=["classifier"])).to(device)
    pairs = list(zip(train_rows[::2], train_rows[1::2]))
    if len(pairs) != 768 or any(left["labels"] or len(right["labels"]) != 1
                                for left, right in pairs):
        raise ValueError("V25 training pairs differ")
    encoded = [tuple(tokenizer(make_input(row), truncation=True,
                               max_length=MAX_LENGTH) for row in pair)
               for pair in pairs]

    def collate(batch):
        return tokenizer.pad([item for pair in batch for item in pair],
                             padding=True, return_tensors="pt")

    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad), lr=LR)
    steps_per_epoch = math.ceil(len(pairs) / PAIRS_PER_BATCH)
    total_steps = EPOCHS * steps_per_epoch
    warmup = max(1, round(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup
        else max(0.0, (total_steps - step) / (total_steps - warmup)))
    output.mkdir(parents=True)
    baseline, _ = fit_char_baseline(train_rows, dev_rows)
    selection = {
        "study": "v25_bilingual_synthetic", "status": "development_selection_before_test_reveal",
        "source_commit": source_commit, "model": MODEL_ID, "base_revision": REVISION,
        "base_files_sha256": BASE_HASHES,
        "test_seed_sha256": record["test_seed_sha256"],
        "test_seed_commitment_sha256": sha256(ROOT / COMMITMENT),
        "train_sha256": record["train_sha256"], "dev_sha256": record["dev_sha256"],
        "taxonomy_sha256": record["taxonomy_sha256"],
        "prompt_sha256": sha256(ROOT / "src/roguard/v25_abstract.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(ROOT / "eval/v25_metrics.py"),
        "protocol_sha256": sha256(ROOT / "docs/V25_SYNTHETIC_PROTOCOL.md"),
        "seed": SEED, "epochs": EPOCHS, "pairs_per_batch": PAIRS_PER_BATCH,
        "learning_rate": LR, "pair_loss_weight": PAIR_WEIGHT,
        "max_length": MAX_LENGTH, "max_prompt_tokens": max(lengths),
        "device": device, "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "char_baseline_dev": baseline,
        "candidates": [],
    }
    for epoch in range(1, EPOCHS + 1):
        loader = DataLoader(encoded, batch_size=PAIRS_PER_BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED + epoch),
                            collate_fn=collate)
        model.train()
        loss_sum = 0.0
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch.to(device)).logits.float()
            labels = torch.tensor([0, 1] * (logits.shape[0] // 2), device=device)
            primary = torch.nn.functional.cross_entropy(logits, labels)
            margins = logits[:, 1] - logits[:, 0]
            ranking = torch.nn.functional.softplus(margins[::2] - margins[1::2]).mean()
            loss = primary + PAIR_WEIGHT * ranking
            if not math.isfinite(float(loss.detach().cpu())):
                raise ValueError("V25 training loss is not finite")
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            if step % 48 == 0:
                print(f"epoch={epoch} step={step}/{steps_per_epoch} mean_loss={loss_sum/step:.4f}",
                      flush=True)
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        scores = score_rows(model, tokenizer, dev_rows, device)
        thresholds = select_thresholds(dev_rows, scores)
        report = evaluate(dev_rows, scores, thresholds)
        selection["candidates"].append({
            "epoch": epoch, "mean_training_loss": loss_sum / len(loader),
            "weight_sha256": sha256(checkpoint / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256(checkpoint / "adapter_config.json"),
            "thresholds": thresholds, "scores": scores, "report": report,
        })
        print(f"epoch={epoch} dev_pairs={report['exact_pairs']}/{report['pairs']} "
              f"worst_cell={min(report['cells'][cell]['exact_pairs'] for cell in CELLS)}/48",
              flush=True)
    selected = max(selection["candidates"], key=_selection_key)
    selection["selected_epoch"] = selected["epoch"]
    selection["selected_thresholds"] = selected["thresholds"]
    selection["selected_weight_sha256"] = selected["weight_sha256"]
    selection_path = output / "development-selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    ARTIFACT.mkdir(parents=True)
    checkpoint = output / f"epoch-{selected['epoch']}"
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(checkpoint / name, ARTIFACT / name)
    shutil.copy2(selection_path, ARTIFACT / "development-selection.json")
    manifest = {
        "schema_version": 1, "study": "v25_bilingual_synthetic",
        "status": "selected_before_test_reveal",
        "scope": "invented_abstract_metadata_only",
        "base_model": MODEL_ID, "base_revision": REVISION,
        "source_commit": source_commit,
        "test_seed_sha256": record["test_seed_sha256"],
        "test_seed_commitment_sha256": sha256(ROOT / COMMITMENT),
        "train_sha256": record["train_sha256"], "dev_sha256": record["dev_sha256"],
        "selected_epoch": selected["epoch"], "thresholds": selected["thresholds"],
        "weight_sha256": selected["weight_sha256"],
        "adapter_config_sha256": selected["adapter_config_sha256"],
        "development_selection_sha256": sha256(ARTIFACT / "development-selection.json"),
        "prompt_sha256": sha256(ROOT / "src/roguard/v25_abstract.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(ROOT / "eval/v25_metrics.py"),
        "protocol_sha256": sha256(ROOT / "docs/V25_SYNTHETIC_PROTOCOL.md"),
        "independent_language_review": False,
        "eligible_for_live_child_message_screening": False,
    }
    (ARTIFACT / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return {"artifact": str(ARTIFACT), "selected_epoch": selected["epoch"],
            "development_pairs": selected["report"]["exact_pairs"],
            "development_total_pairs": selected["report"]["pairs"],
            "test_rows_generated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/bi-mmbert-v25")
    args = parser.parse_args()
    try:
        print(json.dumps(train(args.base_model_path, args.output), indent=2))
    except (OSError, ValueError, TypeError, AssertionError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
