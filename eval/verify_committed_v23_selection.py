"""Recompute V23 development selection and failure from committed raw scores."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.mmbert_study import BASE_HASHES, MODEL_ID, REVISION, verify_adapter_head
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
TRAIN_BATCHES = ("batch-0028", "batch-0031", "batch-0032", "batch-0033",
                 "batch-0034", "batch-0035", "batch-0036")
DEV_BATCHES = ("batch-0037", "batch-0038")
BASELINE_RUNS = {"batch-0037": "ro-mmbert-v20-test-0037.json",
                 "batch-0038": "ro-nli-v21-test-0038.json"}


def d1_rows(root: Path, batch: str) -> list[dict]:
    rows = [row for row in load_approved(root, [batch]) if row["source_kind"] == "message"]
    if len(rows) != 96 or any(left["labels"] or right["labels"] != ["D1"]
                              for left, right in zip(rows[::2], rows[1::2])):
        raise ValueError(f"V23 development cards differ: {batch}")
    return rows


def summarize(rows_by_batch: dict, predictions: dict) -> tuple[dict, dict]:
    reports = {}
    for batch in DEV_BATCHES:
        rows, items = rows_by_batch[batch], predictions[batch]
        reports[batch] = evaluate_pairs(rows, items)
        for half in range(2):
            start = half * 48
            reports[f"{batch}-surface-{half + 1}"] = evaluate_pairs(
                rows[start:start + 48], items[start:start + 48])
    surfaces = [reports[f"{batch}-surface-{half}"]["exact_pairs"]
                for batch in DEV_BATCHES for half in (1, 2)]
    summary = {
        "surface_pairs": surfaces, "worst_surface_pairs": min(surfaces),
        "total_pairs": sum(reports[batch]["exact_pairs"] for batch in DEV_BATCHES),
        "false_reviews": sum(reports[batch]["false_review_on_negatives"]
                             for batch in DEV_BATCHES),
        "positive_recovered": sum(next(item for item in reports[batch]["per_category"]
                                       if item["category"] == "D1")["tp"]
                                  for batch in DEV_BATCHES),
    }
    return reports, summary


def verify(root: Path = ROOT) -> dict:
    root = root.resolve()
    for script in ("verify_mmbert_v20.py", "verify_nli_v21.py"):
        runpy.run_path(str(root / "eval" / script))["verify"](root)
    selection_path = root / "eval/runs/ro-mmbert-v23-dev-selection.json"
    record = json.loads(selection_path.read_text(encoding="utf-8"))
    artifact = root / "models/ro-mmbert-v23-abstract"
    manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
    config = json.loads((artifact / "adapter_config.json").read_text(encoding="utf-8"))
    source = root / "models/ro-mmbert-v16-abstract"
    expected_baseline_hashes = {batch: sha256(root / "eval/runs" / filename)
                                for batch, filename in BASELINE_RUNS.items()}
    if (record.get("status") != "development_selection_only" or
            record.get("study") != "v23_group_dro_worst_surface_d1" or
            record.get("source_commit") != "8f59073305e45614023d14de2dd17fc1d199f6b6" or
            record.get("model") != MODEL_ID or record.get("revision") != REVISION or
            record.get("base_files_sha256") != BASE_HASHES or
            record.get("source_adapter_weight_sha256") != sha256(source / "adapter_model.safetensors") or
            record.get("source_manifest_sha256") != sha256(source / "research.json") or
            record.get("train_batches") != list(TRAIN_BATCHES) or
            record.get("development_batches") != list(DEV_BATCHES) or
            record.get("train_batch_sha256") != {
                batch: sha256(root / f"data/synthetic/{batch}.jsonl") for batch in TRAIN_BATCHES} or
            record.get("development_sha256") != {
                batch: sha256(root / f"data/synthetic/{batch}.jsonl") for batch in DEV_BATCHES} or
            record.get("baseline_run_sha256") != expected_baseline_hashes or
            record.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            record.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py") or
            record.get("study_code_sha256") != sha256(root / "src/roguard/mmbert_study.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_mmbert_v23.py") or
            record.get("protocol_sha256") != sha256(root / "docs/V23_PROTOCOL.md") or
            record.get("seed") != 20261005 or record.get("epochs") != 3 or
            record.get("pairs_per_group_per_epoch") != 24 or
            record.get("pairs_per_batch") != 2 or record.get("learning_rate") != 2e-5 or
            record.get("pair_loss_weight") != 0.5 or
            record.get("positive_class_weight") != 1.5 or
            record.get("group_dro_eta") != 0.05 or
            record.get("max_length") != 256 or
            record.get("total_steps") != 252):
        raise ValueError("V23 development provenance differs")
    rows = {batch: d1_rows(root, batch) for batch in DEV_BATCHES}
    baseline_predictions = {
        batch: json.loads((root / "eval/runs" / filename).read_text(encoding="utf-8"))[
            "predictions"]["v16"] for batch, filename in BASELINE_RUNS.items()}
    _, baseline = summarize(rows, baseline_predictions)
    if record.get("baseline_summary") != baseline or baseline != {
            "surface_pairs": [10, 23, 18, 24], "worst_surface_pairs": 10,
            "total_pairs": 75, "false_reviews": 13, "positive_recovered": 85}:
        raise ValueError("V23 baseline differs from frozen V16 records")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or [item.get("epoch") for item in candidates] != [1, 2, 3]:
        raise ValueError("V23 development epochs are incomplete")
    for candidate in candidates:
        epoch = candidate["epoch"]
        weights = candidate.get("group_weights")
        losses = candidate.get("per_group_mean_loss")
        if (not isinstance(weights, dict) or set(weights) != set(TRAIN_BATCHES) or
                not isinstance(losses, dict) or set(losses) != set(TRAIN_BATCHES) or
                any(type(weights[batch]) not in (int, float) or
                    not math.isfinite(weights[batch]) or weights[batch] <= 0 or
                    type(losses[batch]) not in (int, float) or
                    not math.isfinite(losses[batch]) or losses[batch] < 0
                    for batch in TRAIN_BATCHES) or
                not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)):
            raise ValueError("V23 group-loss or weight record differs")
        path = root / f"eval/runs/ro-mmbert-v23-epoch-{epoch}-dev.json"
        if candidate.get("dev_predictions_sha256") != sha256(path):
            raise ValueError("V23 raw epoch score hash differs")
        data = json.loads(path.read_text(encoding="utf-8"))
        predictions = data.get("predictions")
        if data.get("batches") != list(DEV_BATCHES) or not isinstance(predictions, dict) or set(predictions) != set(DEV_BATCHES):
            raise ValueError("V23 development score stream is incomplete")
        for batch in DEV_BATCHES:
            items = predictions[batch]
            if not isinstance(items, list) or len(items) != 96:
                raise ValueError("V23 development row count differs")
            for row, item in zip(rows[batch], items):
                scores = item.get("scores")
                if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                        item.get("strict_parse") is not True or
                        not isinstance(scores, dict) or set(scores) != {"D1"} or
                        type(scores["D1"]) not in (float, int) or
                        not math.isfinite(scores["D1"]) or not 0 <= scores["D1"] <= 1 or
                        item.get("predicted") != (["D1"] if scores["D1"] >= 0.5 else [])):
                    raise ValueError(f"V23 development score differs: {row['id']}")
        reports, summary = summarize(rows, predictions)
        if (data.get("reports") != reports or data.get("summary") != summary or
                candidate.get("summary") != summary):
            raise ValueError("V23 development metrics differ from raw scores")
    best = max(candidates, key=lambda item: (
        item["summary"]["worst_surface_pairs"], item["summary"]["total_pairs"],
        -item["summary"]["false_reviews"], -item["epoch"]))
    chosen = best["summary"]
    gate = (chosen["worst_surface_pairs"] >= 18 and chosen["total_pairs"] >= 84 and
            chosen["positive_recovered"] >= 90 and chosen["false_reviews"] <= 8 and
            chosen["total_pairs"] > baseline["total_pairs"])
    if (record.get("selected_epoch") != best["epoch"] or
            record.get("selected_weight_sha256") != best["weight_sha256"] or
            record.get("development_gate_passed") != gate or
            manifest.get("artifact_kind") != "experimental_romanian_abstract_card_adapter" or
            manifest.get("study") != "v23_group_dro_worst_surface_d1" or
            manifest.get("base_model") != MODEL_ID or manifest.get("base_revision") != REVISION or
            manifest.get("source_adapter_weight_sha256") != record["source_adapter_weight_sha256"] or
            manifest.get("adapter_weight_sha256") != sha256(artifact / "adapter_model.safetensors") or
            manifest.get("adapter_weight_sha256") != best["weight_sha256"] or
            manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
            manifest.get("selection_record_sha256") != sha256(selection_path) or
            manifest.get("development_scores_sha256") != sha256(
                root / f"eval/runs/ro-mmbert-v23-epoch-{best['epoch']}-dev.json") or
            manifest.get("selected_epoch") != best["epoch"] or
            manifest.get("shared_threshold") != 0.5 or
            manifest.get("development_gate_passed") != gate or
            manifest.get("evaluation_status") != "selected_development_only" or
            manifest.get("prompt_sha256") != record["prompt_sha256"] or
            config.get("base_model_name_or_path") != MODEL_ID or
            config.get("r") != 16 or config.get("lora_alpha") != 32):
        raise ValueError("V23 selected adapter differs from frozen development rule")
    verify_adapter_head(artifact)
    return {"selected_epoch": best["epoch"], "development_summary": chosen,
            "baseline_summary": baseline, "development_gate_passed": gate,
            "selected_weight_sha256": best["weight_sha256"],
            "new_held_out_test_exists": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
