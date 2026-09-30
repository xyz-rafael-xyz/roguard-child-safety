"""Select a category-balanced Qwen checkpoint on the same attested development split."""

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

from select_v4_dev import score

STEPS = (300, 600, 900, 1200)
MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
BASE_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
EXPECTED_BASE_HASHES = {
    "config.json": "574349e5a343236546fda55e4744a76e181f534182d7dc60ff1bad7e7a502849",
    "model.safetensors": "2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "4397cc477eb6d79715ccd2000accd6b3531928f30029665832fa1b255f24d2b9",
}
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
    run_root = root / "checkpoints" / "ro-qwen-v8"
    adapter = run_root / "adapter"
    metadata = json.loads((adapter / "roguard_metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("prompt_version") != "v4" or
            metadata.get("batches") != ["batch-0014", "batch-0015"] or
            metadata.get("iters") != 1200 or
            metadata.get("base_model_id") != MODEL_ID or
            metadata.get("base_revision") != REVISION or
            metadata.get("training_balance_rule") != "three_copies_of_D1_R1_P1_G1_tasks_train_only" or
            metadata.get("train_tasks") != 1728 or
            any(metadata.get("train_tasks_per_category", {}).get(code) != {"da": 144, "nu": 144}
                for code in CATEGORIES) or
            metadata.get("base_files_sha256") != EXPECTED_BASE_HASHES or
            sha256(Path(metadata["base_model_path"]) / "chat_template.jinja") != CHAT_TEMPLATE_SHA256 or
            EXPECTED_BASE_HASHES !=
            {name: sha256(Path(metadata["base_model_path"]) / name) for name in BASE_FILES}):
        raise ValueError("Compositional fit differs from fixed protocol")
    selection_dir = run_root / "selection"
    if selection_dir.exists():
        raise FileExistsError("Development selection already exists")
    selection_dir.mkdir(parents=True)
    reports = []
    for step in STEPS:
        stage = selection_dir / f"step-{step:04d}"
        stage.mkdir()
        weight = adapter / f"{step:07d}_adapters.safetensors"
        shutil.copy2(weight, stage / "adapters.safetensors")
        shutil.copy2(adapter / "adapter_config.json", stage / "adapter_config.json")
        shutil.copy2(adapter / "roguard_metadata.json", stage / "roguard_metadata.json")
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
    if selected.exists():
        raise FileExistsError("Selected adapter already exists")
    shutil.copytree(selection_dir / f"step-{best['step']:04d}", selected)
    selected_metadata = dict(metadata, selected_step=best["step"], selection_batch="batch-0015",
                             selection_rule="max_exact_pairs_then_exact_rows_then_fewer_false_reviews_then_earlier_step")
    (selected / "roguard_metadata.json").write_text(json.dumps(selected_metadata, indent=2) + "\n", encoding="utf-8")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Development selection record already exists")
    output.write_text(json.dumps({
        "status": "qwen_balanced_development_selection", "development_batch": "batch-0015",
        "development_sha256": sha256(root / "data" / "synthetic" / "batch-0015.jsonl"),
        "train_batch_sha256": sha256(root / "data" / "synthetic" / "batch-0014.jsonl"),
        "train_export_sha256": metadata["train_sha256"],
        "dev_export_sha256": metadata["valid_sha256"],
        "prompt_sha256": sha256(root / "src" / "roguard" / "prompt_v4.py"),
        "parser_profile": PARSER_PROFILE,
        "parser_sha256": sha256(root / "src" / "roguard" / "qwen_output.py"),
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "selection_rule": selected_metadata["selection_rule"],
        "reports": reports, "selected_step": best["step"],
        "selected_weight_sha256": sha256(selected / "adapters.safetensors"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
