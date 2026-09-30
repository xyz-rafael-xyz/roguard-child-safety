"""Recompute the preregistered v3 comparison from frozen run outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from roguard.metrics import metrics
from roguard.prompt import make_prompt_v3, parse_codes
from roguard.review import APPLIES_TO, load_approved, sha256, taxonomy_sha256

RUNS = {
    "ro-v3-adapter": "ro-v3-test-0011.json",
    "romistral-base": "romistral-base-v3-0011.json",
    "qwen3-base": "qwen3-base-v3-0011.json",
    "phi4-mini-base": "phi4-mini-base-v3-0011.json",
    "gemma3-base": "gemma3-base-v3-0011.json",
    "mistral7b-base": "mistral7b-base-v3-0011.json",
}
REGISTRY_KEYS = {
    "romistral-base": "romistral7b",
    "qwen3-base": "qwen3_4b",
    "phi4-mini-base": "phi4_mini",
    "gemma3-base": "gemma3_4b",
    "mistral7b-base": "mistral7b",
}


def summarize(rows: list[dict], output: dict, model: str) -> dict:
    details = output["predictions"] if model == "ro-v3-adapter" else output["outputs"]
    if [item["id"] for item in details] != [row["id"] for row in rows]:
        raise ValueError(f"Run IDs or order differ from frozen test: {model}")
    predictions = {}
    parsed_count = 0
    for row, item in zip(rows, details):
        if model == "ro-v3-adapter":
            codes = tuple(item["predicted"])
            parsed = item["strict_parse"]
        else:
            codes = tuple(item["parsed"] or ())
            parsed = item["strict_parse"] and all(row["source_kind"] in APPLIES_TO[code] for code in codes)
        if not parsed:
            codes = ()
        if parsed and any(row["source_kind"] not in APPLIES_TO[code] for code in codes):
            raise ValueError(f"Inapplicable model code: {item['id']}")
        predictions[row["id"]] = codes
        parsed_count += parsed
    exact_rows = sum(predictions[row["id"]] == tuple(row["labels"]) for row in rows)
    exact_pairs = 0
    correct_flips = 0
    false_review_on_negatives = 0
    for negative, positive in zip(rows[::2], rows[1::2]):
        code = positive["labels"][0]
        neg_codes, pos_codes = predictions[negative["id"]], predictions[positive["id"]]
        exact_pairs += neg_codes == () and pos_codes == (code,)
        correct_flips += code not in neg_codes and code in pos_codes
        false_review_on_negatives += bool(neg_codes)
    return {
        "model": model, "parsed_rows": parsed_count, "exact_rows": exact_rows,
        "exact_pairs": exact_pairs, "correct_direction_flips": correct_flips,
        "false_review_on_negatives": false_review_on_negatives,
        "per_category": metrics(rows, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_approved(root, ["batch-0011"])
    if len(rows) != 48 or any(row["split"] != "test" for row in rows):
        raise ValueError("Frozen test contract differs")
    for negative, positive in zip(rows[::2], rows[1::2]):
        if negative["labels"] or len(positive["labels"]) != 1 or negative["source_kind"] != positive["source_kind"]:
            raise ValueError("Expected adjacent negative/positive pairs")
    run_dir = root / "eval" / "runs"
    registry = json.loads((root / "eval" / "models.json").read_text(encoding="utf-8"))
    reports = []
    hashes = {}
    for model, filename in RUNS.items():
        path = run_dir / filename
        output = json.loads(path.read_text(encoding="utf-8"))
        details = output["predictions"] if model == "ro-v3-adapter" else output["outputs"]
        if len(details) != len(rows):
            raise ValueError(f"Run length differs from frozen test: {model}")
        for row, item in zip(rows, details):
            if item["id"] != row["id"] or item["expected"] != row["labels"]:
                raise ValueError(f"Run label or ID differs from frozen test: {model}")
            if model != "ro-v3-adapter":
                prompt_digest = hashlib.sha256(make_prompt_v3(row).encode("utf-8")).hexdigest()
                if (item["prompt_sha256"] != prompt_digest or
                        (None if item["parsed"] is None else tuple(item["parsed"])) != parse_codes(item["raw"]) or
                        item["strict_parse"] != (item["parsed"] is not None)):
                    raise ValueError(f"Base output provenance differs from frozen prompt: {model}")
        if model == "ro-v3-adapter":
            if output.get("batches") != ["batch-0011"]:
                raise ValueError("Adapter run used another test batch")
        else:
            entry = registry[REGISTRY_KEYS[model]]
            if (output.get("batches") != ["batch-0011"] or
                    output.get("prompt_version") != "v3" or output.get("max_tokens") != 24 or
                    output.get("revision") != entry["revision"] or output.get("model") != entry["model"] or
                    output.get("taxonomy_sha256") != {"ro": taxonomy_sha256(root, "ro")}):
                raise ValueError(f"Base run differs from fixed protocol: {model}")
        reports.append(summarize(rows, output, model))
        hashes[filename] = sha256(path)
    adapter = root / "checkpoints" / "ro-v3" / "adapter"
    adapter_metadata = json.loads((adapter / "roguard_metadata.json").read_text(encoding="utf-8"))
    if (adapter_metadata.get("batches") != ["batch-0009", "batch-0010"] or
            adapter_metadata.get("prompt_version") != "v3" or adapter_metadata.get("iters") != 600):
        raise ValueError("Adapter metadata differs from fixed protocol")
    if (adapter_metadata.get("train_sha256") != sha256(root / "checkpoints" / "ro-v3" / "data" / "train.jsonl") or
            adapter_metadata.get("valid_sha256") != sha256(root / "checkpoints" / "ro-v3" / "data" / "valid.jsonl")):
        raise ValueError("Adapter export digests differ from local training data")
    result = {
        "status": "frozen_v3_comparison", "test_batch": "batch-0011",
        "test_sha256": sha256(root / "data" / "synthetic" / "batch-0011.jsonl"),
        "attestation_sha256": sha256(root / "data" / "synthetic" / "batch-0011.review.json"),
        "train_batch_sha256": sha256(root / "data" / "synthetic" / "batch-0009.jsonl"),
        "dev_batch_sha256": sha256(root / "data" / "synthetic" / "batch-0010.jsonl"),
        "train_export_sha256": adapter_metadata["train_sha256"],
        "dev_export_sha256": adapter_metadata["valid_sha256"],
        "base_revision": adapter_metadata["base_revision"],
        "prompt_sha256": sha256(root / "src" / "roguard" / "prompt.py"),
        "adapter_weight_sha256": sha256(adapter / "adapters.safetensors"),
        "run_sha256": hashes, "reports": reports,
        "always_review_reference": {"positive_recall": 1.0, "false_review_on_negatives": 24,
                                     "exact_rows": 24, "exact_pairs": 0},
        "limitation": "Abstract paired cards; repeated rule structures and no real child language. Test is consumed after this comparison.",
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError("Comparison already exists")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
