"""Recompute the frozen v6 comparison from attested test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.review import load_approved, sha256, taxonomy_sha256

from compare_v4 import summarize

RUNS = {
    "ro-v6-adapter": "ro-v6-test-0016.json",
    "ro-v4-adapter": "ro-v4-test-0016.json",
    "romistral-base": "romistral-base-v4-0016.json",
    "qwen3-base": "qwen3-base-v4-0016.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_approved(root, ["batch-0016"])
    if len(rows) != 144 or any(row["split"] != "test" for row in rows):
        raise ValueError("Frozen test differs")
    if any(left["labels"] or len(right["labels"]) != 1 or left["source_kind"] != right["source_kind"]
           for left, right in zip(rows[::2], rows[1::2])):
        raise ValueError("Expected adjacent one-fact test pairs")
    run_dir = root / "eval" / "runs"
    registry = json.loads((root / "eval" / "models.json").read_text(encoding="utf-8"))
    loaded = {}
    for name, filename in RUNS.items():
        run = json.loads((run_dir / filename).read_text(encoding="utf-8"))
        if run.get("batches") != ["batch-0016"]:
            raise ValueError(f"Wrong test batch: {name}")
        if name.endswith("-base"):
            entry = registry["romistral7b" if name == "romistral-base" else "qwen3_4b"]
            if (run.get("model") != entry["model"] or run.get("revision") != entry["revision"] or
                    run.get("prompt_version") != "v4_binary" or run.get("max_tokens_per_task") != 8 or
                    run.get("taxonomy_sha256") != {"ro": taxonomy_sha256(root, "ro")}):
                raise ValueError(f"Base configuration differs: {name}")
        loaded[name] = run
    selection_path = run_dir / "ro-v6-dev-selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = root / "checkpoints" / "ro-v6" / "selected-adapter"
    metadata = json.loads((selected / "roguard_metadata.json").read_text(encoding="utf-8"))
    if (selection.get("development_batch") != "batch-0015" or
            selection.get("development_sha256") != sha256(root / "data" / "synthetic" / "batch-0015.jsonl") or
            selection.get("train_batch_sha256") != sha256(root / "data" / "synthetic" / "batch-0014.jsonl") or
            selection.get("prompt_sha256") != sha256(root / "src" / "roguard" / "prompt_v4.py") or
            metadata.get("batches") != ["batch-0014", "batch-0015"] or
            metadata.get("selected_step") != selection["selected_step"] or
            metadata.get("prompt_version") != "v4" or
            metadata.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            sha256(selected / "adapters.safetensors") != selection["selected_weight_sha256"]):
        raise ValueError("Selected v6 adapter differs from development choice")
    v4_selected = root / "checkpoints" / "ro-v4" / "selected-adapter"
    v4_selection = json.loads((run_dir / "ro-v4-dev-selection.json").read_text(encoding="utf-8"))
    if sha256(v4_selected / "adapters.safetensors") != v4_selection["selected_weight_sha256"]:
        raise ValueError("V4 comparator differs from its development choice")
    for name, path in (("ro-v6-adapter", selected), ("ro-v4-adapter", v4_selected)):
        if loaded[name].get("adapter") != str(path.relative_to(root)) or loaded[name].get("rows") != 144:
            raise ValueError(f"Adapter run path or length differs: {name}")
    reports = []
    for name, run in loaded.items():
        report = summarize(rows, run, "ro-v4-adapter" if name.endswith("-adapter") else name)
        report["model"] = name
        reports.append(report)
    v6 = reports[0]
    success = (v6["exact_pairs"] >= 58 and v6["false_review_on_negatives"] <= 7 and
               all(item["tp"] >= 9 for item in v6["per_category"]) and
               all(v6["exact_pairs"] > item["exact_pairs"] for item in reports[1:]))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Comparison already exists")
    output.write_text(json.dumps({
        "status": "frozen_v6_compositional_comparison", "test_batch": "batch-0016",
        "test_sha256": sha256(root / "data" / "synthetic" / "batch-0016.jsonl"),
        "attestation_sha256": sha256(root / "data" / "synthetic" / "batch-0016.review.json"),
        "selection_record_sha256": sha256(selection_path),
        "selected_weight_sha256": sha256(selected / "adapters.safetensors"),
        "v4_comparator_weight_sha256": sha256(v4_selected / "adapters.safetensors"),
        "prompt_sha256": sha256(root / "src" / "roguard" / "prompt_v4.py"),
        "run_sha256": {filename: sha256(run_dir / filename) for filename in RUNS.values()},
        "reports": reports,
        "always_review_reference": {"exact_rows": 72, "exact_pairs": 0,
                                    "correct_direction_flips": 0, "false_review_on_negatives": 72},
        "never_review_reference": {"exact_rows": 72, "exact_pairs": 0,
                                   "correct_direction_flips": 0, "false_review_on_negatives": 0},
        "preregistered_success": success,
        "limitation": "Compositional symbolic descriptions only; no real child language or deployment validity.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
