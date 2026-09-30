"""Reselect frozen v7 weights on development cards under the amended parser."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

from roguard.model import MLXClassifier
from roguard.qwen_output import PARSER_PROFILE, parse_qwen_binary
from roguard.review import CATEGORIES, load_approved, sha256
from roguard.screen import screen

from select_qwen_dev import BASE_FILES, EXPECTED_BASE_HASHES, MODEL_ID, REVISION, STEPS
from select_v4_dev import score

CHAT_TEMPLATE_SHA256 = "40c21f34cf67d8c760ef72f8ad3ae5afad514299d4b06e91dd9a8d705af7b541"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_approved(root, ["batch-0015"])
    if len(rows) != 96 or any(row["split"] != "dev" for row in rows):
        raise ValueError("Frozen development rows differ")
    source = root / "checkpoints" / "ro-qwen-v7" / "adapter"
    metadata = json.loads((source / "roguard_metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("prompt_version") != "v4" or
            metadata.get("batches") != ["batch-0014", "batch-0015"] or
            metadata.get("iters") != 1200 or metadata.get("train_tasks") != 960 or
            metadata.get("base_model_id") != MODEL_ID or metadata.get("base_revision") != REVISION or
            metadata.get("base_files_sha256") != EXPECTED_BASE_HASHES or
            sha256(Path(metadata["base_model_path"]) / "chat_template.jinja") != CHAT_TEMPLATE_SHA256 or
            EXPECTED_BASE_HASHES !=
            {name: sha256(Path(metadata["base_model_path"]) / name) for name in BASE_FILES}):
        raise ValueError("V7 source weights differ from the frozen fit")
    run_root = root / "checkpoints" / "ro-qwen-v7-wrapper"
    if run_root.exists():
        raise FileExistsError("Wrapper reselection output already exists")
    run_root.mkdir(parents=True)
    reports = []
    for step in STEPS:
        stage = run_root / f"step-{step:04d}"
        stage.mkdir()
        weight = source / f"{step:07d}_adapters.safetensors"
        shutil.copy2(weight, stage / "adapters.safetensors")
        shutil.copy2(source / "adapter_config.json", stage / "adapter_config.json")
        shutil.copy2(source / "roguard_metadata.json", stage / "roguard_metadata.json")
        backend = MLXClassifier(stage, binary_parser=parse_qwen_binary)
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
    selected = run_root / "selected-adapter"
    shutil.copytree(run_root / f"step-{best['step']:04d}", selected)
    selection_rule = "max_exact_pairs_then_exact_rows_then_fewer_false_reviews_then_earlier_step"
    (selected / "roguard_metadata.json").write_text(json.dumps(dict(
        metadata, selected_step=best["step"], selection_batch="batch-0015",
        selection_rule=selection_rule, parser_profile=PARSER_PROFILE,
    ), indent=2) + "\n", encoding="utf-8")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Development selection record already exists")
    output.write_text(json.dumps({
        "status": "qwen_v7_wrapper_development_selection",
        "development_batch": "batch-0015",
        "development_sha256": sha256(root / "data" / "synthetic" / "batch-0015.jsonl"),
        "train_batch_sha256": sha256(root / "data" / "synthetic" / "batch-0014.jsonl"),
        "train_export_sha256": metadata["train_sha256"],
        "dev_export_sha256": metadata["valid_sha256"],
        "prompt_sha256": sha256(root / "src" / "roguard" / "prompt_v4.py"),
        "parser_profile": PARSER_PROFILE,
        "parser_sha256": sha256(root / "src" / "roguard" / "qwen_output.py"),
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "source_fit": "ro-qwen-v7",
        "selection_rule": selection_rule,
        "reports": reports,
        "selected_step": best["step"],
        "selected_weight_sha256": sha256(selected / "adapters.safetensors"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
