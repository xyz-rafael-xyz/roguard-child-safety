"""Recompute the frozen V21 development failure from committed score files."""

from __future__ import annotations

import json
import math
import re
import runpy
from pathlib import Path

from roguard.mmbert_study import verify_adapter_head
from roguard.nli_v21 import BASE_HASHES, FIELDS, MODEL_ID, REVISION, decision
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
TRAIN_BATCHES = ("batch-0028", "batch-0031", "batch-0032", "batch-0033",
                 "batch-0034", "batch-0036")


def verify(root: Path = ROOT) -> dict:
    record_path = root / "eval/runs/ro-nli-v21-dev-selection.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    adapter = root / "models/ro-nli-v21-abstract"
    manifest = json.loads((adapter / "research.json").read_text(encoding="utf-8"))
    if (record.get("study") != "v21_nli_adapted_d1" or
            not re.fullmatch(r"[0-9a-f]{40}", record.get("source_commit", "")) or
            record.get("model") != MODEL_ID or record.get("revision") != REVISION or
            record.get("base_files_sha256") != BASE_HASHES or
            record.get("train_batches") != list(TRAIN_BATCHES) or
            record.get("development_batch") != "batch-0037" or
            record.get("train_pairs") != 264 or
            record.get("seed") != 20261003 or record.get("epochs") != 3 or
            record.get("pair_batch_size") != 2 or record.get("learning_rate") != 3e-5 or
            record.get("max_length") != 256 or
            record.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            record.get("protocol_sha256") != sha256(root / "docs/V21_PROTOCOL.md") or
            record.get("generator_sha256") != sha256(root / "training/generate_nli_transfer_holdout.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_nli_v21.py") or
            record.get("prompt_sha256") != sha256(root / "src/roguard/nli_v21.py") or
            record.get("development_sha256") != sha256(root / "data/synthetic/batch-0037.jsonl") or
            record.get("train_batch_sha256") != {
                batch: sha256(root / f"data/synthetic/{batch}.jsonl") for batch in TRAIN_BATCHES}):
        raise ValueError("Committed V21 development record differs from frozen inputs")
    rows = load_approved(root, ["batch-0037"])
    raw = runpy.run_path(str(root / "training/generate_balanced_joint_holdout.py"))[
        "build_d1_facts"]("batch-0037")
    truth = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in raw]
    if len(rows) != 96 or len(truth) != len(rows) or any(
            row["labels"] != (["D1"] if decision(facts) else [])
            for row, facts in zip(rows, truth)):
        raise ValueError("V21 development states differ from attested labels")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or [item.get("epoch") for item in candidates] != [1, 2, 3]:
        raise ValueError("V21 development candidates are incomplete")
    for candidate in candidates:
        path = root / f"eval/runs/ro-nli-v21-epoch-{candidate['epoch']}-dev.json"
        if sha256(path) != candidate.get("dev_predictions_sha256"):
            raise ValueError("V21 epoch prediction bytes differ")
        dev = json.loads(path.read_text(encoding="utf-8"))
        items = dev.get("predictions")
        if dev.get("batch") != "batch-0037" or not isinstance(items, list) or len(items) != 96:
            raise ValueError("V21 epoch score file is incomplete")
        correct = 0
        for row, facts, item in zip(rows, truth, items):
            if item.get("id") != row["id"] or item.get("field_expected") != facts:
                raise ValueError("V21 epoch field targets differ")
            scores = item.get("field_scores")
            if not isinstance(scores, dict) or set(scores) != set(FIELDS):
                raise ValueError("V21 epoch field scores are incomplete")
            for field in FIELDS:
                value = scores[field]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("V21 epoch field score is invalid")
                correct += (value >= 0.5) == facts[field]
            if item.get("predicted") != (["D1"] if decision({
                    field: scores[field] >= 0.5 for field in FIELDS}) else []):
                raise ValueError("V21 epoch D1 rule differs")
        metrics = evaluate_pairs(rows, items)
        if (candidate.get("field_correct") != correct or dev.get("field_correct") != correct or
                candidate.get("report") != metrics or dev.get("report") != metrics):
            raise ValueError("V21 epoch metrics differ from raw scores")
    best = max(candidates, key=lambda item: (
        item["report"]["exact_pairs"], -item["report"]["false_review_on_negatives"],
        item["field_correct"], -item["epoch"]))
    positive = next(item for item in best["report"]["per_category"] if item["category"] == "D1")
    gate = (best["report"]["exact_pairs"] >= 42 and positive["tp"] >= 42 and
            best["report"]["false_review_on_negatives"] <= 6 and
            best["field_correct"] >= 340)
    if (record.get("selected_epoch") != best["epoch"] or
            record.get("selected_weight_sha256") != best["weight_sha256"] or
            record.get("development_gate_passed") != gate or
            manifest.get("selected_epoch") != best["epoch"] or
            manifest.get("development_gate_passed") != gate or
            manifest.get("selection_record_sha256") != sha256(record_path) or
            manifest.get("development_scores_sha256") != sha256(
                root / f"eval/runs/ro-nli-v21-epoch-{best['epoch']}-dev.json") or
            manifest.get("adapter_weight_sha256") != sha256(adapter / "adapter_model.safetensors") or
            manifest.get("adapter_weight_sha256") != best["weight_sha256"] or
            manifest.get("adapter_config_sha256") != sha256(adapter / "adapter_config.json") or
            manifest.get("prompt_sha256") != sha256(root / "src/roguard/nli_v21.py") or
            manifest.get("base_model") != MODEL_ID or
            manifest.get("base_revision") != REVISION):
        raise ValueError("V21 selected adapter differs from development rule")
    verify_adapter_head(adapter, num_labels=3, require_pooler=True)
    return {"selected_epoch": best["epoch"], "dev_pairs": best["report"]["exact_pairs"],
            "dev_false_reviews": best["report"]["false_review_on_negatives"],
            "field_correct": best["field_correct"], "development_gate_passed": gate,
            "selected_weight_sha256": best["weight_sha256"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
