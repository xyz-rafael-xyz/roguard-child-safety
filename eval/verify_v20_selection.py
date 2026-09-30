"""Recompute V20's class weights, development scores and frozen selection."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.mmbert_study import BASE_HASHES, MODEL_ID, REVISION
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS, decision
from roguard.review import load_approved, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
SELECTION = Path("eval/runs/ro-mmbert-v20-dev-selection.json")
TRAIN_BATCHES = ("batch-0028", "batch-0031", "batch-0032", "batch-0033",
                 "batch-0034", "batch-0036")


def verify(root: Path = ROOT) -> dict:
    record = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    old = runpy.run_path(str(root / "training/v18_facts.py"))
    earlier = [facts for batch in TRAIN_BATCHES[:-1]
               for _, facts in old["load_d1_rows_with_facts"](root, batch)]
    generator = runpy.run_path(str(root / "training/generate_joint_factor_holdout.py"))
    extra = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in generator["build_d1_facts"]("batch-0036")]
    train = earlier + extra
    positives = {field: sum(facts[field] for facts in train) for field in FIELDS}
    weights = {field: (len(train) - count) / count for field, count in positives.items()}
    rows = load_approved(root, ["batch-0035"])
    raw = generator["build_d1_facts"]("batch-0035")
    truth = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in raw]
    if (len(train) != 528 or len(rows) != 48 or len(truth) != len(rows) or
            record.get("status") != "development_selection" or
            record.get("study") != "v20_balanced_joint_d1" or
            record.get("model") != MODEL_ID or record.get("revision") != REVISION or
            record.get("base_files_sha256") != BASE_HASHES or
            record.get("train_batches") != list(TRAIN_BATCHES) or
            record.get("development_batch") != "batch-0035" or
            record.get("train_batch_sha256") != {
                name: sha256(root / f"data/synthetic/{name}.jsonl") for name in TRAIN_BATCHES} or
            record.get("development_sha256") != sha256(root / "data/synthetic/batch-0035.jsonl") or
            record.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            record.get("generator_sha256") != sha256(root / "training/generate_balanced_joint_holdout.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_mmbert_balanced_joint.py") or
            record.get("shared_training_helpers_sha256") != sha256(root / "training/train_mmbert_joint_factors.py") or
            record.get("consumed_training_generator_sha256") != sha256(root / "training/generate_joint_factor_holdout.py") or
            record.get("protocol_sha256") != sha256(root / "docs/V20_PROTOCOL.md") or
            record.get("seed") != 20261002 or record.get("epochs") != 5 or
            record.get("pair_batch_size") != 4 or record.get("train_pairs") != 264 or
            record.get("positive_counts") != positives or
            record.get("positive_class_weights") != weights or
            record.get("learning_rate") != 8e-5 or record.get("max_length") != 256 or
            not isinstance(record.get("candidates"), list) or len(record["candidates"]) != 5):
        raise ValueError("V20 development provenance or training weights differ")
    candidates = []
    for epoch, candidate in enumerate(record["candidates"], 1):
        path = root / f"eval/runs/ro-mmbert-v20-epoch-{epoch}-dev.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        items = saved.get("predictions")
        if (candidate.get("epoch") != epoch or
                candidate.get("dev_predictions_sha256") != sha256(path) or
                type(candidate.get("mean_train_loss")) not in (int, float) or
                not math.isfinite(candidate["mean_train_loss"]) or
                not isinstance(items, list) or len(items) != len(rows) or
                saved.get("batch") != "batch-0035"):
            raise ValueError(f"V20 development epoch differs: {epoch}")
        correct = 0
        for row, facts, item in zip(rows, truth, items):
            scores = item.get("field_scores")
            if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                    item.get("field_expected") != facts or item.get("strict_parse") is not True or
                    not isinstance(scores, dict) or set(scores) != set(FIELDS) or
                    any(type(value) not in (int, float) or not math.isfinite(value) or
                        not 0 <= value <= 1 for value in scores.values()) or
                    row["labels"] != (["D1"] if decision(facts) else [])):
                raise ValueError(f"V20 development field scores differ: {epoch} {row['id']}")
            judgments = {field: scores[field] >= 0.5 for field in FIELDS}
            correct += sum(judgments[field] == facts[field] for field in FIELDS)
            if item.get("predicted") != (["D1"] if decision(judgments) else []):
                raise ValueError(f"V20 development decision differs: {epoch} {row['id']}")
        metrics = evaluate_pairs(rows, items)
        if (saved.get("report") != metrics or saved.get("field_correct") != correct or
                candidate.get("report") != metrics or candidate.get("field_correct") != correct):
            raise ValueError(f"V20 development report differs: {epoch}")
        candidates.append(candidate)
    selected = max(candidates, key=lambda item: (
        item["report"]["exact_pairs"], -item["report"]["false_review_on_negatives"],
        item["field_correct"], -item["epoch"]))
    d1 = selected["report"]["per_category"][0]
    passed = (selected["report"]["exact_pairs"] >= 21 and d1["tp"] >= 21 and
              selected["report"]["false_review_on_negatives"] <= 3 and
              selected["field_correct"] >= 170)
    if (record.get("selected_epoch") != selected["epoch"] or
            record.get("selected_weight_sha256") != selected["weight_sha256"] or
            record.get("development_gate_passed") != passed):
        raise ValueError("V20 development selection differs")
    return {"selected_epoch": selected["epoch"],
            "selected_weight_sha256": selected["weight_sha256"],
            "dev_pairs": selected["report"]["exact_pairs"],
            "dev_false_reviews": selected["report"]["false_review_on_negatives"],
            "field_correct": selected["field_correct"],
            "development_gate_passed": passed}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
