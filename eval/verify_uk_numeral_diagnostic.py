"""Replay the frozen post hoc Ukrainian A1 diagnostic from saved raw output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from roguard.prompt import parse_codes
from roguard.review import APPLIES_TO, sha256

from diagnose_uk_numerals import (
    ORIGINAL_SHA256,
    ROOT,
    SOURCE_SHA256,
    _counts,
    corrected_rows,
    make_prompt,
)

RESULT_SHA256 = "9f12b6415acca383fe667e73d88bc70e9a4f9a1d15ba8556649ec93a1e3c4e9d"
RESULT = Path("eval/runs/uk-numeral-agreement-diagnostic.json")
ORIGINAL = Path("eval/runs/uk-contrast-0001-qwen3-base.json")


def verify(root: Path = ROOT) -> dict:
    rows, changes = corrected_rows(root)
    if sha256(root / ORIGINAL) != ORIGINAL_SHA256 or sha256(root / RESULT) != RESULT_SHA256:
        raise ValueError("Frozen Ukrainian diagnostic inputs or result differ")
    original = json.loads((root / ORIGINAL).read_text(encoding="utf-8"))
    result = json.loads((root / RESULT).read_text(encoding="utf-8"))
    registry = json.loads((root / "eval/models.json").read_text(encoding="utf-8"))["qwen3_4b"]
    selected = [item for item in original["outputs"] if item["id"] in {row["id"] for row in rows}]
    if ([item["id"] for item in selected] != [row["id"] for row in rows] or
            original["model"] != registry["model"] or
            original["revision"] != registry["revision"]):
        raise ValueError("Original A1 run differs")
    expected_metadata = {
        "schema_version": 1,
        "study": "posthoc_consumed_uk_numeral_agreement_only",
        "source_sha256": SOURCE_SHA256,
        "original_run_sha256": ORIGINAL_SHA256,
        "model": registry["model"],
        "revision": registry["revision"],
        "max_tokens": 24,
        "changed_rows": changes,
        "independent_test": False,
        "ukrainian_language_validated": False,
        "real_child_language_accuracy_established": False,
    }
    if (any(result.get(key) != value for key, value in expected_metadata.items()) or
            set(result) != {*expected_metadata, "mlx_lm_version", "original_a1",
                            "corrected_a1", "decision_changes", "outputs"} or
            result["mlx_lm_version"] != "0.31.3"):
        raise ValueError("Diagnostic metadata differs")
    outputs = result["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(rows):
        raise ValueError("Diagnostic output count differs")
    for row, item in zip(rows, outputs):
        if set(item) != {"id", "expected", "predicted", "strict_parse", "raw", "prompt_sha256"}:
            raise ValueError("Diagnostic row shape differs")
        prompt_hash = hashlib.sha256(make_prompt(row).encode("utf-8")).hexdigest()
        codes = parse_codes(item["raw"])
        valid = codes is not None and all(row["source_kind"] in APPLIES_TO[code] for code in codes)
        prediction = list(codes or ()) if valid else []
        if (item["id"] != row["id"] or item["expected"] != row["labels"] or
                item["prompt_sha256"] != prompt_hash or item["strict_parse"] is not valid or
                item["predicted"] != prediction):
            raise ValueError(f"Diagnostic row differs: {row['id']}")
    original_counts = _counts(selected)
    corrected_counts = _counts(outputs)
    changed = [item["id"] for item, prior in zip(outputs, selected)
               if item["predicted"] != prior["predicted"] or
               item["strict_parse"] != prior["strict_parse"]]
    if (result["original_a1"] != original_counts or
            result["corrected_a1"] != corrected_counts or
            result["decision_changes"] != changed or
            original_counts != {"parsed_cards": 8, "exact_cards": 4, "exact_pairs": 0} or
            corrected_counts != original_counts or changed):
        raise ValueError("Diagnostic summary differs")
    return {"original_a1": original_counts, "corrected_a1": corrected_counts,
            "decision_changes": changed}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
