"""Fit class-balanced latent facts for the prospective V27 study."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval.v26_metrics import CELLS, evaluate, select_thresholds
from roguard.mmbert_study import BASE_HASHES, MAX_LENGTH, MODEL_ID, REVISION, verify_base
from roguard.review import sha256, taxonomy_sha256
from roguard.v26_facts import make_input, score_rows, truth_vector
from training.generate_v27_balanced import (ARTIFACT, COMMITMENT, DATA, PUBLIC_SEEDS,
                                         ROOT, build_rows)
from training.train_v25_synthetic import fit_char_baseline
from training.v27_weights import fact_value_weights

SEED = 20261003
EPOCHS = 8
PAIRS_PER_BATCH = 4
LR = 3e-5
OUTCOME_WEIGHT = 0.5
PAIR_WEIGHT = 0.25
OUTPUT = ROOT / "checkpoints/bi-mmbert-v27"


def verify_inputs(root: Path) -> tuple[dict, list[dict], list[dict], str]:
    root = root.resolve()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Commit V27 code, protocol, train/dev rows, and seed seal before fitting")
    record = json.loads((root / COMMITMENT).read_text(encoding="utf-8"))
    if (record.get("status") != "sealed_same_origin_test_not_yet_generated" or
            record.get("test_rows_at_registration") != 0 or
            record.get("generator_sha256") != sha256(root / "training/generate_v27_balanced.py") or
            record.get("source_dependency_sha256") != {
                name: sha256(root / name) for name in (
                    "training/generate_v25_synthetic.py",
                    "training/generate_v26_facts.py",
                    "training/v25_ro.py", "training/v25_uk.py",
                    "training/v26_ro.py", "training/v26_uk.py")
            } or record.get("taxonomy_sha256") != {
                lang: taxonomy_sha256(root, lang) for lang in ("ro", "uk")
            } or (root / DATA / "test.jsonl").exists()):
        raise ValueError("V27 input commitment or frozen taxonomy differs")
    rows = {}
    for split, seed in PUBLIC_SEEDS.items():
        path = root / DATA / f"{split}.jsonl"
        if sha256(path) != record[f"{split}_sha256"]:
            raise ValueError(f"V27 {split} hash differs")
        actual = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if actual != build_rows(split, seed):
            raise ValueError(f"V27 {split} does not reproduce its generator")
        rows[split] = actual
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return record, rows["train"], rows["dev"], commit


def _selection_key(candidate: dict) -> tuple:
    cells = candidate["report"]["cells"]
    return (min(cells[cell]["exact_pairs"] for cell in CELLS),
            candidate["report"]["exact_pairs"],
            sum(cells[cell]["exact_cards"] for cell in CELLS),
            candidate["fact_report"]["correct"], -candidate["epoch"])


def fact_report(rows: list[dict], vectors: list[list[float]]) -> dict:
    counts = {name: [0, 0] for name in (
        "source_role_minor", "safety_or_support_anchor",
        "indirect_or_repeated_support_pattern", "explicit_support_request",
        "applicable", "passed")}
    if len(rows) != len(vectors):
        raise ValueError("V27 fact scores and rows differ")
    for row, scores in zip(rows, vectors):
        truth, mask = truth_vector(row)
        if len(scores) != 6:
            raise ValueError("V27 needs six fact scores")
        for index, name in enumerate(counts):
            if mask[index]:
                counts[name][1] += 1
                counts[name][0] += int((scores[index] >= 0.5) == bool(truth[index]))
    return {"correct": sum(item[0] for item in counts.values()),
            "total": sum(item[1] for item in counts.values()),
            "by_fact": {name: {"correct": value[0], "total": value[1]}
                        for name, value in counts.items()}}


def train(base: Path, output: Path = OUTPUT) -> dict:
    record, train_rows, dev_rows, source_commit = verify_inputs(ROOT)
    base, output = base.resolve(), output.resolve()
    if output.exists() or (ROOT / ARTIFACT).exists():
        raise FileExistsError("Preserve any V27 fitting output or selected artifact")
    verify_base(base)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    lengths = [len(tokenizer(make_input(row), add_special_tokens=True)["input_ids"])
               for row in train_rows + dev_rows]
    if max(lengths) > MAX_LENGTH:
        raise ValueError("V27 prompt exceeds the registered 256-token cap")
    foundation = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=6, use_safetensors=False, local_files_only=True)
    model = get_peft_model(foundation, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["Wqkv", "Wo"], modules_to_save=["classifier"])).to(device)
    pairs = list(zip(train_rows[::2], train_rows[1::2]))
    if len(pairs) != 768 or any(left["labels"] or len(right["labels"]) != 1
                                for left, right in pairs):
        raise ValueError("V27 training pairs differ")
    encoded = [tuple((tokenizer(make_input(row), truncation=True,
                               max_length=MAX_LENGTH), *truth_vector(row),
                      row["source_kind"] == "message") for row in pair)
               for pair in pairs]
    positive_values, negative_values = fact_value_weights(train_rows)
    positive_weights = torch.tensor(positive_values, dtype=torch.float32, device=device)
    negative_weights = torch.tensor(negative_values, dtype=torch.float32, device=device)

    def collate(batch):
        flat = [item for pair in batch for item in pair]
        return (tokenizer.pad([item[0] for item in flat], padding=True, return_tensors="pt"),
                torch.tensor([item[1] for item in flat], dtype=torch.float32),
                torch.tensor([item[2] for item in flat], dtype=torch.float32),
                torch.tensor([item[3] for item in flat], dtype=torch.bool))

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
    baseline["report"] = evaluate(dev_rows, baseline["scores"], baseline["thresholds"])
    selection = {
        "study": "v27_bilingual_balanced_fact_synthetic",
        "status": "development_selection_before_test_reveal",
        "source_commit": source_commit, "base_model": MODEL_ID,
        "base_revision": REVISION, "base_files_sha256": BASE_HASHES,
        "test_seed_sha256": record["test_seed_sha256"],
        "test_seed_commitment_sha256": sha256(ROOT / COMMITMENT),
        "train_sha256": record["train_sha256"], "dev_sha256": record["dev_sha256"],
        "taxonomy_sha256": record["taxonomy_sha256"],
        "prompt_sha256": sha256(ROOT / "src/roguard/v26_facts.py"),
        "loader_sha256": sha256(ROOT / "src/roguard/v27_balanced.py"),
        "weighting_sha256": sha256(ROOT / "training/v27_weights.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(ROOT / "eval/v26_metrics.py"),
        "protocol_sha256": sha256(ROOT / "docs/V27_BALANCED_PROTOCOL.md"),
        "seed": SEED, "epochs": EPOCHS, "pairs_per_batch": PAIRS_PER_BATCH,
        "learning_rate": LR, "outcome_loss_weight": OUTCOME_WEIGHT,
        "pair_loss_weight": PAIR_WEIGHT, "max_length": MAX_LENGTH,
        "fact_loss_weighting": "per_fact_inverse_class_frequency_v1",
        "positive_fact_weights": positive_weights.cpu().tolist(),
        "negative_fact_weights": negative_weights.cpu().tolist(),
        "max_prompt_tokens": max(lengths), "device": device,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "char_baseline_dev": baseline, "candidates": [],
    }
    for epoch in range(1, EPOCHS + 1):
        loader = DataLoader(encoded, batch_size=PAIRS_PER_BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED + epoch),
                            collate_fn=collate)
        model.train()
        loss_sum = 0.0
        for step, (batch, facts, mask, is_message) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch.to(device)).logits.float()
            facts, mask, is_message = facts.to(device), mask.to(device), is_message.to(device)
            value_weights = torch.where(facts.bool(), positive_weights, negative_weights)
            fact_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits, facts, reduction="none") * value_weights * mask).sum() / mask.sum()
            probabilities = torch.sigmoid(logits)
            d1 = probabilities[:, 0] * probabilities[:, 1] * (
                1 - (1 - probabilities[:, 2]) * (1 - probabilities[:, 3]))
            s1 = probabilities[:, 4] * (1 - probabilities[:, 5])
            outcome = torch.where(is_message, d1, s1).clamp(1e-6, 1 - 1e-6)
            labels = torch.tensor([0, 1] * (len(outcome) // 2),
                                  device=device, dtype=torch.float32)
            outcome_loss = torch.nn.functional.binary_cross_entropy(outcome, labels)
            ranking = torch.nn.functional.softplus(
                torch.logit(outcome[::2]) - torch.logit(outcome[1::2])).mean()
            loss = fact_loss + OUTCOME_WEIGHT * outcome_loss + PAIR_WEIGHT * ranking
            if not math.isfinite(float(loss.detach().cpu())):
                raise ValueError("V27 training loss is not finite")
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            if step % 48 == 0:
                print(f"epoch={epoch} step={step}/{steps_per_epoch} mean_loss={loss_sum/step:.4f}",
                      flush=True)
        checkpoint = output / f"epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        scores, vectors = score_rows(model, tokenizer, dev_rows, device)
        thresholds = select_thresholds(dev_rows, scores)
        report = evaluate(dev_rows, scores, thresholds)
        facts_report = fact_report(dev_rows, vectors)
        selection["candidates"].append({
            "epoch": epoch, "mean_training_loss": loss_sum / len(loader),
            "weight_sha256": sha256(checkpoint / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256(checkpoint / "adapter_config.json"),
            "thresholds": thresholds, "scores": scores,
            "fact_scores": vectors, "report": report, "fact_report": facts_report,
        })
        print(f"epoch={epoch} dev_pairs={report['exact_pairs']}/{report['pairs']} "
              f"worst_cell={min(report['cells'][cell]['exact_pairs'] for cell in CELLS)}/48 "
              f"facts={facts_report['correct']}/{facts_report['total']}", flush=True)
    selected = max(selection["candidates"], key=_selection_key)
    selection["selected_epoch"] = selected["epoch"]
    selection["selected_thresholds"] = selected["thresholds"]
    selection["selected_weight_sha256"] = selected["weight_sha256"]
    development_eligible = (selected["report"]["exact_pairs"] >= 168 and
                            all(selected["report"]["cells"][cell]["exact_pairs"] >= 40
                                for cell in CELLS))
    selection["development_eligible_for_test_reveal"] = development_eligible
    selection_path = output / "development-selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    artifact_path = ROOT / ARTIFACT
    artifact_path.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(output / f"epoch-{selected['epoch']}" / name, artifact_path / name)
    shutil.copy2(selection_path, artifact_path / "development-selection.json")
    manifest = {
        "schema_version": 1, "study": "v27_bilingual_balanced_fact_synthetic",
        "status": "selected_before_test_reveal", "scope": "invented_abstract_metadata_only",
        "base_model": MODEL_ID, "base_revision": REVISION,
        "source_commit": source_commit,
        "test_seed_sha256": record["test_seed_sha256"],
        "test_seed_commitment_sha256": sha256(ROOT / COMMITMENT),
        "train_sha256": record["train_sha256"], "dev_sha256": record["dev_sha256"],
        "selected_epoch": selected["epoch"], "thresholds": selected["thresholds"],
        "weight_sha256": selected["weight_sha256"],
        "adapter_config_sha256": selected["adapter_config_sha256"],
        "development_selection_sha256": sha256(artifact_path / "development-selection.json"),
        "development_eligible_for_test_reveal": development_eligible,
        "prompt_sha256": sha256(ROOT / "src/roguard/v26_facts.py"),
        "loader_sha256": sha256(ROOT / "src/roguard/v27_balanced.py"),
        "weighting_sha256": sha256(ROOT / "training/v27_weights.py"),
        "trainer_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(ROOT / "eval/v26_metrics.py"),
        "protocol_sha256": sha256(ROOT / "docs/V27_BALANCED_PROTOCOL.md"),
        "independent_language_review": False,
        "eligible_for_live_child_message_screening": False,
    }
    (artifact_path / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return {"artifact": str(ARTIFACT), "selected_epoch": selected["epoch"],
            "development_pairs": selected["report"]["exact_pairs"],
            "development_worst_cell": min(selected["report"]["cells"][cell]["exact_pairs"]
                                          for cell in CELLS),
            "development_eligible_for_test_reveal": development_eligible,
            "test_rows_generated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.base_model_path), indent=2))


if __name__ == "__main__":
    main()
