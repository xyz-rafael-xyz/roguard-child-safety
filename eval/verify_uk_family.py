"""Replay saved post hoc Ukrainian model-family decisions and summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roguard.prompt import parse_codes
from roguard.review import APPLIES_TO, sha256, taxonomy_sha256

from diagnose_uk_family import (
    CORRECTION_SHA256,
    MODELS,
    PROMPT_SHA256,
    REGISTRY_SHA256,
    ROOT,
    SOURCE_SHA256,
    category_counts,
    corrected_probe,
    make_prompt,
)
from diagnose_uk_numerals import ORIGINAL_SHA256

RESULTS = {
    "phi4_mini": ("uk-family-phi4-mini-corrected.json",
                  "66e8a59de67e183eecec445de9c2d0dcf2bbe642b0a953e8db46ed462b0138ae"),
    "gemma3_4b": ("uk-family-gemma3-4b-corrected.json",
                   "3e2498f32a0a7d6023f2d7553daee2b7a60ce415f200a1facfd10b7681103416"),
    "mistral7b": ("uk-family-mistral7b-corrected.json",
                   "d5f0af692c65b70c9e28ec507189fa50e6705fed1f18c7d00e6770b58d2ae627"),
}


def _summary(rows: list[dict], outputs: list[dict]) -> dict:
    pairs = list(zip(outputs[::2], outputs[1::2]))
    return {
        "parsed_rows": sum(item["strict_parse"] for item in outputs),
        "exact_rows": sum(item["strict_parse"] and item["predicted"] == item["expected"]
                          for item in outputs),
        "exact_pairs": sum(left["strict_parse"] and right["strict_parse"] and
                           left["predicted"] == [] and right["predicted"] == right["expected"]
                           for left, right in pairs),
        "correct_direction_flips": sum(left["strict_parse"] and right["strict_parse"] and
                                       right["expected"][0] not in left["predicted"] and
                                       right["expected"][0] in right["predicted"]
                                       for left, right in pairs),
        "category_counts": category_counts(rows, outputs),
    }


def verify(root: Path = ROOT) -> dict:
    rows = corrected_probe(root)
    registry = json.loads((root / "eval/models.json").read_text(encoding="utf-8"))
    if set(RESULTS) != set(MODELS):
        raise ValueError("Ukrainian diagnostic model set differs")
    report = {}
    for name in MODELS:
        filename, expected_hash = RESULTS[name]
        path = root / "eval/runs" / filename
        if sha256(path) != expected_hash:
            raise ValueError(f"Frozen Ukrainian model result differs: {name}")
        result = json.loads(path.read_text(encoding="utf-8"))
        model = registry[name]
        metadata = {
            "schema_version": 1,
            "study": "posthoc_consumed_uk_model_family_only",
            "independent_test": False,
            "ukrainian_language_validated": False,
            "real_child_language_accuracy_established": False,
            "source_sha256": SOURCE_SHA256,
            "corrected_a1_diagnostic_sha256": CORRECTION_SHA256,
            "registry_sha256": REGISTRY_SHA256,
            "prompt_code_sha256": PROMPT_SHA256,
            "taxonomy_sha256": taxonomy_sha256(root, "uk"),
            "model_key": name,
            "model": model["model"],
            "revision": model["revision"],
            "mlx_lm_version": "0.31.3",
            "max_tokens": 24,
            "rows": len(rows),
            "pairs": len(rows) // 2,
        }
        if (not isinstance(result, dict) or
                set(result) != {*metadata, "parsed_rows", "exact_rows", "exact_pairs",
                                "correct_direction_flips", "category_counts", "outputs"} or
                any(result.get(key) != value for key, value in metadata.items()) or
                not isinstance(result["outputs"], list) or len(result["outputs"]) != len(rows)):
            raise ValueError(f"Ukrainian diagnostic metadata differs: {name}")
        outputs = result["outputs"]
        for row, item in zip(rows, outputs):
            if (not isinstance(item, dict) or set(item) != {
                    "id", "pair_id", "variant", "expected", "predicted", "strict_parse",
                    "raw", "prompt_sha256"}):
                raise ValueError(f"Ukrainian diagnostic row shape differs: {name}")
            prompt_hash = hashlib.sha256(make_prompt(row).encode("utf-8")).hexdigest()
            codes = parse_codes(item["raw"])
            valid = codes is not None and all(row["source_kind"] in APPLIES_TO[code] for code in codes)
            prediction = list(codes or ()) if valid else []
            if (item["id"] != row["id"] or item["pair_id"] != row["pair_id"] or
                    item["variant"] != row["variant"] or item["expected"] != row["labels"] or
                    item["prompt_sha256"] != prompt_hash or item["strict_parse"] is not valid or
                    item["predicted"] != prediction):
                raise ValueError(f"Ukrainian diagnostic row differs: {name} {row['id']}")
        summary = _summary(rows, outputs)
        if any(result[key] != value for key, value in summary.items()):
            raise ValueError(f"Ukrainian diagnostic summary differs: {name}")
        report[name] = {key: summary[key] for key in (
            "parsed_rows", "exact_rows", "exact_pairs", "correct_direction_flips")}
    original_path = root / "eval/runs/uk-contrast-0001-qwen3-base.json"
    if sha256(original_path) != ORIGINAL_SHA256:
        raise ValueError("Historical Qwen comparator differs")
    original = json.loads(original_path.read_text(encoding="utf-8"))["outputs"]
    corrected = json.loads((root / "eval/runs/uk-numeral-agreement-diagnostic.json").read_text(
        encoding="utf-8"))["outputs"]
    replacements = {item["id"]: item for item in corrected}
    qwen = []
    for row, item in zip(rows, original):
        if item["id"] != row["id"] or item["expected"] != row["labels"]:
            raise ValueError("Historical Qwen comparator order differs")
        selected = replacements.get(item["id"], item)
        qwen.append({"id": row["id"], "expected": row["labels"],
                     "predicted": selected["predicted"],
                     "strict_parse": selected["strict_parse"]})
    qwen_summary = _summary(rows, qwen)
    if qwen_summary["parsed_rows"] != 46 or qwen_summary["exact_rows"] != 21 or qwen_summary["exact_pairs"] != 0:
        raise ValueError("Historical Qwen comparator summary differs")
    report["qwen3_4b_historical_corrected_a1"] = {key: qwen_summary[key] for key in (
        "parsed_rows", "exact_rows", "exact_pairs", "correct_direction_flips")}
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
