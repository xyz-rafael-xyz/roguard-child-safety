"""Recompute the preregistered Qwen transfer comparison on batch 0017."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.review import load_approved, sha256, taxonomy_sha256
from compare_v4 import summarize

RUNS = {
    "ro-qwen-v7-adapter": "ro-qwen-v7-test-0017.json",
    "ro-v6-adapter": "ro-v6-test-0017.json",
    "qwen3-base": "qwen3-base-v4-0017.json",
}
BASE_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
EXPECTED_BASE_HASHES = {
    "config.json": "574349e5a343236546fda55e4744a76e181f534182d7dc60ff1bad7e7a502849",
    "model.safetensors": "2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "4397cc477eb6d79715ccd2000accd6b3531928f30029665832fa1b255f24d2b9",
}


def _selection(root: Path, name: str, selection_name: str) -> tuple[dict, Path]:
    run_dir = root / "eval" / "runs"
    selection = json.loads((run_dir / selection_name).read_text(encoding="utf-8"))
    adapter = root / "checkpoints" / name / "selected-adapter"
    metadata = json.loads((adapter / "roguard_metadata.json").read_text(encoding="utf-8"))
    if (selection.get("development_batch") != "batch-0015" or
            selection.get("development_sha256") != sha256(root / "data" / "synthetic" / "batch-0015.jsonl") or
            selection.get("train_batch_sha256") != sha256(root / "data" / "synthetic" / "batch-0014.jsonl") or
            selection.get("prompt_sha256") != sha256(root / "src" / "roguard" / "prompt_v4.py") or
            metadata.get("batches") != ["batch-0014", "batch-0015"] or
            metadata.get("selected_step") != selection.get("selected_step") or
            metadata.get("prompt_version") != "v4" or
            metadata.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            sha256(adapter / "adapters.safetensors") != selection.get("selected_weight_sha256")):
        raise ValueError(f"Adapter differs from development choice: {name}")
    return metadata, adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_approved(root, ["batch-0017"])
    if len(rows) != 144 or any(row["split"] != "test" for row in rows):
        raise ValueError("Frozen transfer test differs")
    if any(left["labels"] or len(right["labels"]) != 1 or left["source_kind"] != right["source_kind"]
           for left, right in zip(rows[::2], rows[1::2])):
        raise ValueError("Expected adjacent one-fact test pairs")
    run_dir = root / "eval" / "runs"
    registry = json.loads((root / "eval" / "models.json").read_text(encoding="utf-8"))
    loaded = {name: json.loads((run_dir / filename).read_text(encoding="utf-8"))
              for name, filename in RUNS.items()}
    if any(run.get("batches") != ["batch-0017"] for run in loaded.values()):
        raise ValueError("Wrong test batch")
    base = loaded["qwen3-base"]
    entry = registry["qwen3_4b"]
    if (base.get("model") != entry["model"] or base.get("revision") != entry["revision"] or
            base.get("prompt_version") != "v4_binary" or base.get("max_tokens_per_task") != 8 or
            base.get("taxonomy_sha256") != {"ro": taxonomy_sha256(root, "ro")}):
        raise ValueError("Prompt-only Qwen configuration differs")
    metadata, selected = _selection(root, "ro-qwen-v7", "ro-qwen-v7-dev-selection.json")
    _, old_selected = _selection(root, "ro-v6", "ro-v6-dev-selection.json")
    if (metadata.get("base_model_id") != entry["model"] or
            metadata.get("base_revision") != entry["revision"] or
            metadata.get("base_files_sha256") != EXPECTED_BASE_HASHES or
            EXPECTED_BASE_HASHES !=
            {name: sha256(Path(metadata["base_model_path"]) / name) for name in BASE_FILES}):
        raise ValueError("Qwen base differs from training provenance")
    for name, path in (("ro-qwen-v7-adapter", selected), ("ro-v6-adapter", old_selected)):
        if loaded[name].get("adapter") != str(path.relative_to(root)) or loaded[name].get("rows") != 144:
            raise ValueError(f"Adapter run path or length differs: {name}")
    reports = []
    for name, run in loaded.items():
        report = summarize(rows, run, "ro-v4-adapter" if name.endswith("-adapter") else name)
        report["model"] = name
        reports.append(report)
    candidate = reports[0]
    success = (candidate["exact_pairs"] >= 58 and
               candidate["false_review_on_negatives"] <= 7 and
               all(item["tp"] >= 9 for item in candidate["per_category"]) and
               all(candidate["exact_pairs"] > item["exact_pairs"] for item in reports[1:]))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Comparison already exists")
    output.write_text(json.dumps({
        "status": "frozen_qwen_transfer_comparison", "test_batch": "batch-0017",
        "test_sha256": sha256(root / "data" / "synthetic" / "batch-0017.jsonl"),
        "attestation_sha256": sha256(root / "data" / "synthetic" / "batch-0017.review.json"),
        "selection_record_sha256": sha256(run_dir / "ro-qwen-v7-dev-selection.json"),
        "v6_selection_record_sha256": sha256(run_dir / "ro-v6-dev-selection.json"),
        "selected_weight_sha256": sha256(selected / "adapters.safetensors"),
        "v6_comparator_weight_sha256": sha256(old_selected / "adapters.safetensors"),
        "prompt_sha256": sha256(root / "src" / "roguard" / "prompt_v4.py"),
        "run_sha256": {filename: sha256(run_dir / filename) for filename in RUNS.values()},
        "reports": reports,
        "always_review_reference": {"exact_rows": 72, "exact_pairs": 0,
                                    "correct_direction_flips": 0, "false_review_on_negatives": 72},
        "never_review_reference": {"exact_rows": 72, "exact_pairs": 0,
                                   "correct_direction_flips": 0, "false_review_on_negatives": 0},
        "preregistered_success": success,
        "limitation": "Symbolic descriptions only; no real child language or deployment validity.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
