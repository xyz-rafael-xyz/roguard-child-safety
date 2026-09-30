"""Replay V24 selection, metrics, and hashes from committed development scores."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.mmbert_study import verify_adapter_head
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
TRAIN_BATCHES = ("batch-0028", "batch-0031", "batch-0032", "batch-0033",
                 "batch-0034", "batch-0035", "batch-0036")
DEV_BATCHES = ("batch-0037", "batch-0038")
MODEL_ID = "dumitrescustefan/bert-base-romanian-cased-v1"
REVISION = "37fb0ffb4bc4f7c4cde429626775685fb18f234f"
BASE_HASHES = {
    "config.json": "53d69f53eb2fa0a9e95ae77508ad527dc1ce658a74968a565fbae5cc29495542",
    "model.safetensors": "c2471091c9b2613f671fc46f71023b10d8d3dee5e1b1fe2c6c4d3fce1b4b8508",
    "vocab.txt": "1c8630d8abddcb7d36a51f32a25b72084dd12439bb223ee25cca7e81643acef9",
    "tokenizer_config.json": "adac4eb1158c23ecf85b5215105c1d0bcc42e34d6a9f82d637018f4b7f0b1aa2",
}


def _rows(root: Path, batch: str) -> list[dict]:
    rows = [row for row in load_approved(root, [batch]) if row["source_kind"] == "message"]
    if len(rows) != 96 or any(left["labels"] or right["labels"] != ["D1"]
                              for left, right in zip(rows[::2], rows[1::2])):
        raise ValueError("V24 development cards differ")
    return rows


def _summarize(rows_by_batch: dict, predictions: dict) -> tuple[dict, dict]:
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


def _cutoff_diagnostic(predictions: dict) -> dict:
    """Post-hoc ceiling on consumed cards only; never select a release cutoff."""
    surfaces = [predictions[batch][half * 48:(half + 1) * 48]
                for batch in DEV_BATCHES for half in (0, 1)]
    scores = sorted({item["scores"]["D1"] for surface in surfaces for item in surface})
    cutoffs = [0.0, 1.0, *scores,
               *((left + right) / 2 for left, right in zip(scores, scores[1:]))]
    best = None
    for cutoff in cutoffs:
        counts = [sum(left["scores"]["D1"] < cutoff <= right["scores"]["D1"]
                      for left, right in zip(surface[::2], surface[1::2]))
                  for surface in surfaces]
        key = (min(counts), sum(counts), -cutoff)
        if best is None or key > best[0]:
            best = key, cutoff, counts
    assert best is not None
    return {"scope": "post_hoc_consumed_development_only",
            "best_worst_surface_pairs": best[0][0],
            "surface_pairs_at_oracle_cutoff": best[2],
            "oracle_cutoff": best[1],
            "no_new_test_or_validated_cutoff": True}


def verify(root: Path = ROOT) -> dict:
    root = root.resolve()
    runpy.run_path(str(root / "eval/verify_committed_v23_selection.py"))["verify"](root)
    selection = root / "eval/runs/ro-bert-v24-dev-selection.json"
    record = json.loads(selection.read_text(encoding="utf-8"))
    artifact = root / "models/ro-bert-v24-abstract"
    manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
    config = json.loads((artifact / "adapter_config.json").read_text(encoding="utf-8"))
    if (record.get("status") != "development_selection_only" or
            record.get("study") != "v24_native_romanian_bert_d1" or
            record.get("source_commit") != "1f4f55361a11a5d9c8b56b8cfc32e3d113471482" or
            record.get("model") != MODEL_ID or record.get("revision") != REVISION or
            record.get("base_files_sha256") != BASE_HASHES or
            record.get("train_batches") != list(TRAIN_BATCHES) or
            record.get("development_batches") != list(DEV_BATCHES) or
            record.get("train_batch_sha256") != {
                batch: sha256(root / f"data/synthetic/{batch}.jsonl") for batch in TRAIN_BATCHES} or
            record.get("development_sha256") != {
                batch: sha256(root / f"data/synthetic/{batch}.jsonl") for batch in DEV_BATCHES} or
            record.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            record.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py") or
            record.get("predictor_sha256") != sha256(root / "src/roguard/mmbert_study.py") or
            record.get("data_selector_sha256") != sha256(root / "training/train_mmbert_v23.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_robert_v24.py") or
            record.get("protocol_sha256") != sha256(root / "docs/V24_PROTOCOL.md") or
            record.get("v23_selection_sha256") != sha256(
                root / "eval/runs/ro-mmbert-v23-dev-selection.json") or
            record.get("v16_baseline_summary") != {
                "surface_pairs": [10, 23, 18, 24], "worst_surface_pairs": 10,
                "total_pairs": 75, "false_reviews": 13, "positive_recovered": 85} or
            record.get("seed") != 20261006 or record.get("epochs") != 6 or
            record.get("pairs_per_group_per_epoch") != 24 or
            record.get("pairs_per_batch") != 2 or
            record.get("learning_rate_lora") != 2e-4 or
            record.get("learning_rate_classifier") != 1e-3 or
            record.get("pair_loss_weight") != 0.5 or
            record.get("max_length") != 256 or
            record.get("max_prompt_tokens") != 134 or
            record.get("trainable_parameters") != 591362 or
            record.get("total_steps") != 504):
        raise ValueError("V24 development provenance differs")
    v23 = json.loads((root / "eval/runs/ro-mmbert-v23-dev-selection.json").read_text(
        encoding="utf-8"))["candidates"]
    v23_best = max(v23, key=lambda item: (
        item["summary"]["worst_surface_pairs"], item["summary"]["total_pairs"],
        -item["summary"]["false_reviews"], -item["epoch"]))["summary"]
    if record.get("v23_development_summary") != v23_best:
        raise ValueError("V24 V23 comparator differs")
    rows = {batch: _rows(root, batch) for batch in DEV_BATCHES}
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or [item.get("epoch") for item in candidates] != list(range(1, 7)):
        raise ValueError("V24 epoch records are incomplete")
    selected_predictions = None
    for candidate in candidates:
        epoch = candidate["epoch"]
        if (type(candidate.get("mean_train_loss")) not in (int, float) or
                not math.isfinite(candidate["mean_train_loss"]) or
                candidate["mean_train_loss"] < 0):
            raise ValueError("V24 training loss differs")
        path = root / f"eval/runs/ro-bert-v24-epoch-{epoch}-dev.json"
        if candidate.get("dev_predictions_sha256") != sha256(path):
            raise ValueError("V24 raw epoch score hash differs")
        data = json.loads(path.read_text(encoding="utf-8"))
        predictions = data.get("predictions")
        if data.get("batches") != list(DEV_BATCHES) or not isinstance(predictions, dict) or set(predictions) != set(DEV_BATCHES):
            raise ValueError("V24 development score stream differs")
        for batch in DEV_BATCHES:
            items = predictions[batch]
            if not isinstance(items, list) or len(items) != 96:
                raise ValueError("V24 development row count differs")
            for row, item in zip(rows[batch], items):
                scores = item.get("scores")
                if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                        item.get("strict_parse") is not True or
                        not isinstance(scores, dict) or set(scores) != {"D1"} or
                        type(scores["D1"]) not in (float, int) or
                        not math.isfinite(scores["D1"]) or not 0 <= scores["D1"] <= 1 or
                        item.get("predicted") != (["D1"] if scores["D1"] >= 0.5 else [])):
                    raise ValueError("V24 development score row differs")
        reports, summary = _summarize(rows, predictions)
        if (data.get("reports") != reports or data.get("summary") != summary or
                candidate.get("summary") != summary):
            raise ValueError("V24 metrics differ from raw scores")
        if epoch == record.get("selected_epoch"):
            selected_predictions = predictions
    best = max(candidates, key=lambda item: (
        item["summary"]["worst_surface_pairs"], item["summary"]["total_pairs"],
        -item["summary"]["false_reviews"], -item["epoch"]))
    summary = best["summary"]
    gate = (summary["worst_surface_pairs"] >= 18 and summary["total_pairs"] >= 84 and
            summary["positive_recovered"] >= 90 and summary["false_reviews"] <= 8 and
            summary["total_pairs"] > 75)
    if (record.get("selected_epoch") != best["epoch"] or
            record.get("selected_weight_sha256") != best["weight_sha256"] or
            record.get("development_gate_passed") != gate or
            manifest.get("study") != "v24_native_romanian_bert_d1" or
            manifest.get("base_model") != MODEL_ID or manifest.get("base_revision") != REVISION or
            manifest.get("base_files_sha256") != BASE_HASHES or
            manifest.get("adapter_weight_sha256") != sha256(artifact / "adapter_model.safetensors") or
            manifest.get("adapter_weight_sha256") != best["weight_sha256"] or
            manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
            manifest.get("selection_record_sha256") != sha256(selection) or
            manifest.get("development_scores_sha256") != sha256(
                root / f"eval/runs/ro-bert-v24-epoch-{best['epoch']}-dev.json") or
            manifest.get("selected_epoch") != best["epoch"] or
            manifest.get("shared_threshold") != 0.5 or
            manifest.get("development_gate_passed") != gate or
            manifest.get("evaluation_status") != "selected_development_only" or
            manifest.get("prompt_sha256") != record["prompt_sha256"] or
            config.get("base_model_name_or_path") != MODEL_ID or
            config.get("r") != 16 or config.get("lora_alpha") != 32 or
            set(config.get("target_modules", ())) != {"query", "value"}):
        raise ValueError("V24 selected adapter differs from frozen rule")
    verify_adapter_head(artifact)
    if selected_predictions is None:
        raise ValueError("V24 selected predictions are missing")
    return {"selected_epoch": best["epoch"], "development_summary": summary,
            "baseline_summary": record["v16_baseline_summary"],
            "development_gate_passed": gate,
            "post_hoc_cutoff_diagnostic": _cutoff_diagnostic(selected_predictions),
            "new_held_out_test_exists": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
