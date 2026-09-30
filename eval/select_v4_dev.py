"""Choose one saved binary adapter using development pairs only."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

from roguard.model import MLXClassifier
from roguard.review import CATEGORIES, load_approved, sha256
from roguard.screen import screen

STEPS = (100, 200, 300, 400, 500, 600)


def score(rows: list[dict], predictions: dict[str, tuple[str, ...]], parsed: int, step: int) -> dict:
    exact_rows = sum(predictions[row["id"]] == tuple(row["labels"]) for row in rows)
    exact_pairs = sum(predictions[left["id"]] == () and
                      predictions[right["id"]] == tuple(right["labels"])
                      for left, right in zip(rows[::2], rows[1::2]))
    false_review = sum(bool(predictions[row["id"]]) for row in rows[::2])
    return {"step": step, "parsed_rows": parsed, "exact_rows": exact_rows,
            "exact_pairs": exact_pairs, "false_review_on_negatives": false_review}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_approved(root, ["batch-0010"])
    if len(rows) != 36 or any(row["split"] != "dev" for row in rows):
        raise ValueError("Frozen development rows differ")
    adapter = root / "checkpoints" / "ro-v4" / "adapter"
    metadata = json.loads((adapter / "roguard_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("prompt_version") != "v4" or metadata.get("batches") != ["batch-0009", "batch-0010"] or metadata.get("iters") != 600:
        raise ValueError("Binary fit differs from fixed protocol")
    selection_dir = root / "checkpoints" / "ro-v4" / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for step in STEPS:
        stage = selection_dir / f"step-{step:04d}"
        stage.mkdir(exist_ok=False)
        weight = adapter / f"{step:07d}_adapters.safetensors"
        shutil.copy2(weight, stage / "adapters.safetensors")
        shutil.copy2(adapter / "adapter_config.json", stage / "adapter_config.json")
        shutil.copy2(adapter / "roguard_metadata.json", stage / "roguard_metadata.json")
        backend = MLXClassifier(stage)
        predictions = {}
        parsed = 0
        for row in rows:
            try:
                labels = screen(row["text"], language="ro", source_kind=row["source_kind"],
                                backend=backend, thresholds={code: 0.5 for code in CATEGORIES}).labels
                parsed += 1
            except ValueError as exc:
                if "Model output could not be parsed" not in str(exc):
                    raise
                labels = ()
            predictions[row["id"]] = labels
        report = score(rows, predictions, parsed, step)
        report["weight_sha256"] = sha256(weight)
        reports.append(report)
        print(report, flush=True)
        del backend
        gc.collect()
        import mlx.core as mx
        mx.metal.clear_cache()
    best = max(reports, key=lambda item: (item["exact_pairs"], item["exact_rows"],
                                          -item["false_review_on_negatives"], -item["step"]))
    selected = root / "checkpoints" / "ro-v4" / "selected-adapter"
    if selected.exists():
        raise FileExistsError("Selected adapter already exists")
    shutil.copytree(selection_dir / f"step-{best['step']:04d}", selected)
    selected_metadata = dict(metadata, selected_step=best["step"], selection_batch="batch-0010",
                             selection_rule="max_exact_pairs_then_exact_rows_then_fewer_false_reviews_then_earlier_step")
    (selected / "roguard_metadata.json").write_text(json.dumps(selected_metadata, indent=2) + "\n", encoding="utf-8")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Development selection record already exists")
    output.write_text(json.dumps({"status": "development_selection", "development_batch": "batch-0010",
                                  "development_sha256": sha256(root / "data" / "synthetic" / "batch-0010.jsonl"),
                                  "train_export_sha256": metadata["train_sha256"],
                                  "dev_export_sha256": metadata["valid_sha256"],
                                  "selection_rule": selected_metadata["selection_rule"],
                                  "reports": reports, "selected_step": best["step"],
                                  "selected_weight_sha256": sha256(selected / "adapters.safetensors")},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
