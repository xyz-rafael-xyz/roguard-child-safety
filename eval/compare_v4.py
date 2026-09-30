"""Recompute the frozen binary comparison from held-out predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from roguard.metrics import metrics
from roguard.prompt_v4 import applicable_codes, make_prompt_v4, parse_binary
from roguard.review import load_approved, sha256, taxonomy_sha256

RUNS = {
    "ro-v4-adapter": "ro-v4-test-0012.json",
    "romistral-base": "romistral-base-v4-0012.json",
    "qwen3-base": "qwen3-base-v4-0012.json",
    "phi4-mini-base": "phi4-mini-base-v4-0012.json",
    "gemma3-base": "gemma3-base-v4-0012.json",
    "mistral7b-base": "mistral7b-base-v4-0012.json",
}
REGISTRY_KEYS = {
    "romistral-base": "romistral7b", "qwen3-base": "qwen3_4b",
    "phi4-mini-base": "phi4_mini", "gemma3-base": "gemma3_4b",
    "mistral7b-base": "mistral7b",
}


def summarize(rows: list[dict], output: dict, model: str) -> dict:
    details = output["predictions"] if model == "ro-v4-adapter" else output["outputs"]
    if len(details) != len(rows):
        raise ValueError(f"Run length differs: {model}")
    predictions = {}
    parsed_rows = 0
    for row, item in zip(rows, details):
        if item["id"] != row["id"] or item["expected"] != row["labels"]:
            raise ValueError(f"IDs or expected labels differ: {model}")
        codes = tuple(item["predicted"])
        if any(code not in applicable_codes(row["source_kind"]) for code in codes):
            raise ValueError(f"Inapplicable prediction: {model}:{row['id']}")
        if model != "ro-v4-adapter":
            expected_digests = {code: hashlib.sha256(make_prompt_v4(row, code).encode("utf-8")).hexdigest()
                                for code in applicable_codes(row["source_kind"])}
            answers = {code: parse_binary(item["raw"][code]) for code in expected_digests}
            valid = all(answer is not None for answer in answers.values())
            expected_codes = tuple(code for code, answer in answers.items() if answer) if valid else ()
            if item["prompt_sha256"] != expected_digests or item["strict_parse"] != valid or codes != expected_codes:
                raise ValueError(f"Raw response or prompt provenance differs: {model}:{row['id']}")
        if not item["strict_parse"] and codes:
            raise ValueError(f"Unparsed prediction must abstain: {model}:{row['id']}")
        parsed_rows += bool(item["strict_parse"])
        predictions[row["id"]] = codes
    exact_rows = sum(predictions[row["id"]] == tuple(row["labels"]) for row in rows)
    exact_pairs = sum(predictions[left["id"]] == () and
                      predictions[right["id"]] == tuple(right["labels"])
                      for left, right in zip(rows[::2], rows[1::2]))
    flips = sum(right["labels"][0] not in predictions[left["id"]] and
                right["labels"][0] in predictions[right["id"]]
                for left, right in zip(rows[::2], rows[1::2]))
    false_review = sum(bool(predictions[row["id"]]) for row in rows[::2])
    return {"model": model, "parsed_rows": parsed_rows, "exact_rows": exact_rows,
            "exact_pairs": exact_pairs, "correct_direction_flips": flips,
            "false_review_on_negatives": false_review,
            "per_category": metrics(rows, predictions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_approved(root, ["batch-0012"])
    if len(rows) != 48 or any(row["split"] != "test" for row in rows):
        raise ValueError("Frozen test differs")
    if any(left["labels"] or len(right["labels"]) != 1 or left["source_kind"] != right["source_kind"]
           for left, right in zip(rows[::2], rows[1::2])):
        raise ValueError("Expected adjacent one-factor pairs")
    registry = json.loads((root / "eval" / "models.json").read_text(encoding="utf-8"))
    run_dir = root / "eval" / "runs"
    reports = []
    hashes = {}
    for model, filename in RUNS.items():
        path = run_dir / filename
        output = json.loads(path.read_text(encoding="utf-8"))
        if output.get("batches") != ["batch-0012"]:
            raise ValueError(f"Wrong test batch: {model}")
        if model != "ro-v4-adapter":
            entry = registry[REGISTRY_KEYS[model]]
            if (output.get("model") != entry["model"] or output.get("revision") != entry["revision"] or
                    output.get("prompt_version") != "v4_binary" or output.get("max_tokens_per_task") != 8 or
                    output.get("taxonomy_sha256") != {"ro": taxonomy_sha256(root, "ro")}):
                raise ValueError(f"Base configuration differs: {model}")
        reports.append(summarize(rows, output, model))
        hashes[filename] = sha256(path)
    selected = root / "checkpoints" / "ro-v4" / "selected-adapter"
    metadata = json.loads((selected / "roguard_metadata.json").read_text(encoding="utf-8"))
    choice_path = run_dir / "ro-v4-dev-selection.json"
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    if (metadata.get("prompt_version") != "v4" or metadata.get("iters") != 600 or
            metadata.get("batches") != ["batch-0009", "batch-0010"] or
            metadata.get("selected_step") != choice["selected_step"] or
            sha256(selected / "adapters.safetensors") != choice["selected_weight_sha256"] or
            metadata.get("train_sha256") != sha256(root / "checkpoints" / "ro-v4" / "data" / "train.jsonl") or
            metadata.get("valid_sha256") != sha256(root / "checkpoints" / "ro-v4" / "data" / "valid.jsonl")):
        raise ValueError("Selected adapter differs from development choice")
    result = {
        "status": "frozen_v4_binary_comparison", "test_batch": "batch-0012",
        "test_sha256": sha256(root / "data" / "synthetic" / "batch-0012.jsonl"),
        "attestation_sha256": sha256(root / "data" / "synthetic" / "batch-0012.review.json"),
        "train_batch_sha256": sha256(root / "data" / "synthetic" / "batch-0009.jsonl"),
        "dev_batch_sha256": sha256(root / "data" / "synthetic" / "batch-0010.jsonl"),
        "train_export_sha256": metadata["train_sha256"], "dev_export_sha256": metadata["valid_sha256"],
        "prompt_sha256": sha256(root / "src" / "roguard" / "prompt_v4.py"),
        "selected_step": choice["selected_step"], "selection_record_sha256": sha256(choice_path),
        "adapter_weight_sha256": sha256(selected / "adapters.safetensors"),
        "run_sha256": hashes, "reports": reports,
        "always_review_reference": {"exact_rows": 24, "exact_pairs": 0,
                                     "correct_direction_flips": 0, "false_review_on_negatives": 24},
        "limitation": "Symbolic one-factor cards with recurring policy structure; no authentic child language or real-world validity.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Comparison already exists")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
