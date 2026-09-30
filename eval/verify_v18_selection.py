"""Recompute V18 development selection without loading a model."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS, decision
from roguard.review import sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
SELECTION = Path("eval/runs/ro-mmbert-v18-dev-selection.json")
load_d1_rows_with_facts = runpy.run_path(str(ROOT / "training/v18_facts.py"))[
    "load_d1_rows_with_facts"]


def verify(root: Path = ROOT) -> dict:
    record = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    cards = load_d1_rows_with_facts(root, "batch-0033")
    candidates = record.get("candidates")
    train = ("batch-0028", "batch-0031", "batch-0032")
    if (record.get("status") != "development_selection" or
            record.get("study") != "v18_factorized_d1" or
            record.get("model") != MODEL_ID or record.get("revision") != REVISION or
            record.get("train_batches") != list(train) or
            record.get("development_batch") != "batch-0033" or
            record.get("train_batch_sha256") != {
                batch: sha256(root / f"data/synthetic/{batch}.jsonl") for batch in train} or
            record.get("development_sha256") != sha256(root / "data/synthetic/batch-0033.jsonl") or
            record.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            record.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v18.py") or
            record.get("facts_sha256") != sha256(root / "training/v18_facts.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_mmbert_factors.py") or
            record.get("protocol_sha256") != sha256(root / "docs/V18_PROTOCOL.md") or
            record.get("prefit_amendment_sha256") != sha256(root / "docs/V18_PREFIT_AMENDMENT.md") or
            record.get("source_adapter_weight_sha256") != sha256(
                root / "models/ro-mmbert-v16-abstract/adapter_model.safetensors") or
            record.get("factor_task_count") != 1152 or
            not isinstance(candidates, list) or len(candidates) != 3 or
            sorted(item.get("epoch") for item in candidates) != [1, 2, 3]):
        raise ValueError("V18 selection input or protocol differs")
    rows = [row for row, _ in cards]
    for entry in candidates:
        epoch = entry["epoch"]
        path = root / f"eval/runs/ro-mmbert-v18-epoch-{epoch}-dev.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        items = saved.get("predictions")
        if (entry.get("dev_predictions_sha256") != sha256(path) or
                saved.get("batch") != "batch-0033" or
                not isinstance(items, list) or len(items) != len(cards)):
            raise ValueError("V18 development score file differs")
        correct = 0
        for (row, facts), item in zip(cards, items):
            scores = item.get("factor_scores")
            if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                    item.get("strict_parse") is not True or
                    item.get("factor_expected") != facts or
                    not isinstance(scores, dict) or set(scores) != set(FIELDS) or
                    any(type(value) not in (int, float) or not math.isfinite(value) or
                        not 0 <= value <= 1 for value in scores.values()) or
                    item.get("predicted") != (["D1"] if decision({
                        field: scores[field] >= 0.5 for field in FIELDS}) else [])):
                raise ValueError("Invalid V18 factor development score")
            correct += sum((scores[field] >= 0.5) == facts[field] for field in FIELDS)
        report = evaluate_pairs(rows, items)
        if (saved.get("report") != report or saved.get("factor_correct") != correct or
                entry.get("report") != report or entry.get("factor_correct") != correct):
            raise ValueError("V18 development metrics differ")
    best = max(candidates, key=lambda entry: (
        entry["report"]["exact_pairs"],
        -entry["report"]["false_review_on_negatives"],
        entry["factor_correct"], -entry["epoch"]))
    metric = next(item for item in best["report"]["per_category"] if item["category"] == "D1")
    gate = metric["tp"] >= 21 and best["report"]["false_review_on_negatives"] <= 3
    if (record.get("selected_epoch") != best["epoch"] or
            record.get("selected_weight_sha256") != best["weight_sha256"] or
            record.get("development_gate_passed") != gate):
        raise ValueError("V18 fixed development selection differs")
    return {"selected_epoch": best["epoch"],
            "selected_weight_sha256": best["weight_sha256"],
            "dev_exact_pairs": best["report"]["exact_pairs"],
            "dev_false_reviews": best["report"]["false_review_on_negatives"],
            "development_gate_passed": gate}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
