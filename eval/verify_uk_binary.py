"""Replay the saved Ukrainian binary-prompt development diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roguard.review import APPLIES_TO, CATEGORIES, sha256, taxonomy_sha256

from diagnose_uk_binary import MODELS, ROOT, make_binary_prompt
from diagnose_uk_family import (CORRECTION_SHA256, REGISTRY_SHA256,
                                SOURCE_SHA256, category_counts, corrected_probe)

RESULTS = {
    "qwen3_4b": ("uk-binary-qwen3-4b-corrected.json",
                  "582bee940e50bb458911687d0b1e363b3e4e07faf085afab369611e449ecc476"),
    "phi4_mini": ("uk-binary-phi4-mini-corrected.json",
                  "f298ad8f0a69691e34ad5f4e838c50532737cc5d30f04f5688d408570e906052"),
    "gemma3_4b": ("uk-binary-gemma3-4b-corrected.json",
                   "c5082bd5b8e3b86b5ac5baace1d6d853c6f98ef0a3e15f743aca817b2e69d0c5"),
    "mistral7b": ("uk-binary-mistral7b-corrected.json",
                   "8be96fc3411895a222de1dda7ce18e2a05a4e3f72edfe1b7d09b61a6eaff88ca"),
}


def verify(root: Path = ROOT) -> dict:
    rows = corrected_probe(root)
    registry = json.loads((root / "eval/models.json").read_text(encoding="utf-8"))
    if set(RESULTS) != set(MODELS):
        raise ValueError("Ukrainian binary diagnostic model set differs")
    summaries = {}
    for name in MODELS:
        filename, expected_hash = RESULTS[name]
        path = root / "eval/runs" / filename
        if sha256(path) != expected_hash:
            raise ValueError(f"Frozen Ukrainian binary result differs: {name}")
        result = json.loads(path.read_text(encoding="utf-8"))
        model = registry[name]
        metadata = {
            "schema_version": 1,
            "study": "posthoc_consumed_uk_binary_prompt_only",
            "independent_test": False,
            "ukrainian_language_validated": False,
            "real_child_language_accuracy_established": False,
            "source_sha256": SOURCE_SHA256,
            "corrected_a1_diagnostic_sha256": CORRECTION_SHA256,
            "registry_sha256": REGISTRY_SHA256,
            "taxonomy_sha256": taxonomy_sha256(root, "uk"),
            "model_key": name,
            "model": model["model"],
            "revision": model["revision"],
            "mlx_lm_version": "0.31.3",
            "max_tokens": 8,
            "rows": 48,
            "pairs": 24,
        }
        summaries_keys = {"parsed_tasks", "total_tasks", "parsed_rows", "exact_rows",
                          "exact_pairs", "correct_direction_flips", "category_counts"}
        if (not isinstance(result, dict) or
                set(result) != {*metadata, *summaries_keys, "outputs"} or
                any(result.get(key) != value for key, value in metadata.items()) or
                not isinstance(result["outputs"], list) or len(result["outputs"]) != len(rows)):
            raise ValueError(f"Ukrainian binary metadata differs: {name}")
        outputs = result["outputs"]
        for row, item in zip(rows, outputs):
            if (not isinstance(item, dict) or set(item) != {
                    "id", "pair_id", "variant", "expected", "predicted", "strict_parse", "tasks"} or
                    not isinstance(item["tasks"], list)):
                raise ValueError(f"Ukrainian binary row shape differs: {name}")
            codes = [code for code in CATEGORIES if row["source_kind"] in APPLIES_TO[code]]
            if len(item["tasks"]) != len(codes):
                raise ValueError(f"Ukrainian binary task count differs: {name} {row['id']}")
            values = []
            for code, task in zip(codes, item["tasks"]):
                if not isinstance(task, dict) or set(task) != {
                        "category", "raw", "value", "prompt_sha256"}:
                    raise ValueError(f"Ukrainian binary task shape differs: {name}")
                prompt_hash = hashlib.sha256(make_binary_prompt(row, code).encode("utf-8")).hexdigest()
                parsed = task["raw"].strip()
                value = int(parsed) if parsed in ("0", "1") else None
                if (task["category"] != code or task["prompt_sha256"] != prompt_hash or
                        task["value"] != value):
                    raise ValueError(f"Ukrainian binary task differs: {name} {row['id']} {code}")
                values.append(value)
            valid = all(value is not None for value in values)
            predicted = [code for code, value in zip(codes, values) if value == 1] if valid else []
            if (item["id"] != row["id"] or item["pair_id"] != row["pair_id"] or
                    item["variant"] != row["variant"] or item["expected"] != row["labels"] or
                    item["strict_parse"] is not valid or item["predicted"] != predicted):
                raise ValueError(f"Ukrainian binary row differs: {name} {row['id']}")
        pairs = list(zip(outputs[::2], outputs[1::2]))
        summary = {
            "parsed_tasks": sum(task["value"] is not None for row in outputs for task in row["tasks"]),
            "total_tasks": sum(len(row["tasks"]) for row in outputs),
            "parsed_rows": sum(row["strict_parse"] for row in outputs),
            "exact_rows": sum(row["strict_parse"] and row["predicted"] == row["expected"]
                              for row in outputs),
            "exact_pairs": sum(left["strict_parse"] and right["strict_parse"] and
                               left["predicted"] == [] and right["predicted"] == right["expected"]
                               for left, right in pairs),
            "correct_direction_flips": sum(left["strict_parse"] and right["strict_parse"] and
                                           right["expected"][0] not in left["predicted"] and
                                           right["expected"][0] in right["predicted"]
                                           for left, right in pairs),
            "category_counts": category_counts(rows, outputs),
        }
        if summary["total_tasks"] != 64 or any(result[key] != value for key, value in summary.items()):
            raise ValueError(f"Ukrainian binary summary differs: {name}")
        summaries[name] = {key: summary[key] for key in (
            "parsed_tasks", "parsed_rows", "exact_rows", "exact_pairs",
            "correct_direction_flips")}
    return summaries


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
