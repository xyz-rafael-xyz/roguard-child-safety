"""Recompute frozen V19 development selection from saved four-field scores."""

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
SELECTION = Path("eval/runs/ro-mmbert-v19-dev-selection.json")


def verify(root: Path = ROOT) -> dict:
    record = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    generator = runpy.run_path(str(root / "training/generate_joint_factor_holdout.py"))
    rows = load_approved(root, ["batch-0035"])
    raw = generator["build_d1_facts"]("batch-0035")
    facts = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in raw]
    if (len(rows) != 48 or len(facts) != len(rows) or
            record.get("status") != "development_selection" or
            record.get("study") != "v19_joint_factorized_d1" or
            record.get("model") != MODEL_ID or record.get("revision") != REVISION or
            record.get("base_files_sha256") != BASE_HASHES or
            record.get("train_batches") != ["batch-0028", "batch-0031", "batch-0032",
                                             "batch-0033", "batch-0034"] or
            record.get("development_batch") != "batch-0035" or
            record.get("train_batch_sha256") != {
                name: sha256(root / f"data/synthetic/{name}.jsonl")
                for name in record["train_batches"]} or
            record.get("development_sha256") != sha256(root / "data/synthetic/batch-0035.jsonl") or
            record.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            record.get("generator_sha256") != sha256(root / "training/generate_joint_factor_holdout.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_mmbert_joint_factors.py") or
            record.get("protocol_sha256") != sha256(root / "docs/V19_PROTOCOL.md") or
            record.get("seed") != 20261001 or record.get("epochs") != 3 or
            record.get("pair_batch_size") != 4 or record.get("train_pairs") != 216 or
            record.get("learning_rate") != 5e-5 or record.get("max_length") != 256 or
            not isinstance(record.get("candidates"), list) or len(record["candidates"]) != 3):
        raise ValueError("V19 development provenance differs")
    candidates = []
    for epoch, candidate in enumerate(record["candidates"], 1):
        path = root / f"eval/runs/ro-mmbert-v19-epoch-{epoch}-dev.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        items = saved.get("predictions")
        if (candidate.get("epoch") != epoch or
                candidate.get("dev_predictions_sha256") != sha256(path) or
                type(candidate.get("mean_train_loss")) not in (int, float) or
                not math.isfinite(candidate["mean_train_loss"]) or
                not isinstance(items, list) or len(items) != len(rows) or
                saved.get("batch") != "batch-0035"):
            raise ValueError(f"V19 development epoch differs: {epoch}")
        correct = 0
        for row, truth, item in zip(rows, facts, items):
            scores = item.get("field_scores")
            if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                    item.get("field_expected") != truth or item.get("strict_parse") is not True or
                    not isinstance(scores, dict) or set(scores) != set(FIELDS) or
                    any(type(value) not in (int, float) or not math.isfinite(value) or
                        not 0 <= value <= 1 for value in scores.values())):
                raise ValueError(f"V19 development field scores differ: {epoch} {row['id']}")
            judgments = {field: scores[field] >= 0.5 for field in FIELDS}
            correct += sum(judgments[field] == truth[field] for field in FIELDS)
            if item.get("predicted") != (["D1"] if decision(judgments) else []):
                raise ValueError(f"V19 development decision differs: {epoch} {row['id']}")
        report = evaluate_pairs(rows, items)
        if (saved.get("report") != report or saved.get("field_correct") != correct or
                candidate.get("report") != report or candidate.get("field_correct") != correct):
            raise ValueError(f"V19 development report differs: {epoch}")
        candidates.append(candidate)
    selected = max(candidates, key=lambda item: (
        item["report"]["exact_pairs"], -item["report"]["false_review_on_negatives"],
        item["field_correct"], -item["epoch"]))
    positive = next(item for item in selected["report"]["per_category"]
                    if item["category"] == "D1")
    passed = (selected["report"]["exact_pairs"] >= 21 and positive["tp"] >= 21 and
              selected["report"]["false_review_on_negatives"] <= 3 and
              selected["field_correct"] >= 170)
    if (record.get("selected_epoch") != selected["epoch"] or
            record.get("selected_weight_sha256") != selected["weight_sha256"] or
            record.get("development_gate_passed") != passed):
        raise ValueError("V19 selected checkpoint or development gate differs")
    return {"selected_epoch": selected["epoch"],
            "selected_weight_sha256": selected["weight_sha256"],
            "dev_pairs": selected["report"]["exact_pairs"],
            "dev_false_reviews": selected["report"]["false_review_on_negatives"],
            "field_correct": selected["field_correct"],
            "development_gate_passed": passed}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
