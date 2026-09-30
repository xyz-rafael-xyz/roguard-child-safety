"""Post hoc Ukrainian numeral-agreement sensitivity check on a consumed probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from roguard.prompt import parse_codes
from roguard.review import APPLIES_TO, sha256

from run_panel import load_generator
from uk_contrast_suite import build_rows
from uk_prompt_probe import make_prompt

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "87e974bf932620f71159c8b4f17cb02dbc9fa2c1c145656771aa6cbc40b28ef2"
ORIGINAL_SHA256 = "0b2650d5d2de6f760b4b30706f290a573bde7b3f11db8830d09223e462bf7104"
REPLACEMENTS = {"54 слів": "54 слова", "62 слів": "62 слова",
                "63 слів": "63 слова", "92 слів": "92 слова"}
CHANGED_IDS = {"uk-contrast-0001-09-0", "uk-contrast-0001-09-1",
               "uk-contrast-0001-11-1", "uk-contrast-0001-12-0",
               "uk-contrast-0001-12-1"}


def corrected_rows(root: Path) -> tuple[list[dict], list[dict]]:
    source = root / "data/synthetic/uk-contrast-0001.jsonl"
    manifest = json.loads(source.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if (sha256(source) != SOURCE_SHA256 or manifest.get("sha256") != SOURCE_SHA256 or
            manifest.get("training_eligible") is not False):
        raise ValueError("Frozen Ukrainian probe differs from its registered bytes")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if rows != build_rows():
        raise ValueError("Frozen Ukrainian probe differs from its generator")
    selected = [row for row in rows if row["id"].split("-")[-2] in {"09", "10", "11", "12"}]
    if len(selected) != 8 or any(row["source_kind"] != "response" or
                                row["labels"] != (["A1"] if row["variant"] else [])
                                for row in selected):
        raise ValueError("Unexpected A1 probe shape")
    corrected, changes = [], []
    for row in selected:
        text = row["text"]
        substitutions = 0
        for old, new in REPLACEMENTS.items():
            substitutions += text.count(old)
            text = text.replace(old, new)
        if substitutions:
            changes.append({"id": row["id"], "occurrences": substitutions})
        corrected.append({**row, "text": text})
    if {change["id"] for change in changes} != CHANGED_IDS or sum(
            change["occurrences"] for change in changes) != 7:
        raise ValueError("Ukrainian correction set differs from the registered diagnostic")
    return corrected, changes


def _counts(outputs: list[dict]) -> dict:
    exact_cards = sum(item["predicted"] == item["expected"] and item["strict_parse"]
                      for item in outputs)
    exact_pairs = sum(left["strict_parse"] and right["strict_parse"] and
                      left["predicted"] == [] and right["predicted"] == ["A1"]
                      for left, right in zip(outputs[::2], outputs[1::2]))
    return {"parsed_cards": sum(item["strict_parse"] for item in outputs),
            "exact_cards": exact_cards, "exact_pairs": exact_pairs}


def run_diagnostic(root: Path, output: Path) -> dict:
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Diagnostic output already exists")
    rows, changes = corrected_rows(root)
    original_path = root / "eval/runs/uk-contrast-0001-qwen3-base.json"
    if sha256(original_path) != ORIGINAL_SHA256:
        raise ValueError("Original Ukrainian Qwen run differs from registered bytes")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    original_rows = [item for item in original["outputs"] if item["id"] in
                     {row["id"] for row in rows}]
    if len(original_rows) != 8 or [item["id"] for item in original_rows] != [row["id"] for row in rows]:
        raise ValueError("Original A1 prediction order differs")
    registry = json.loads((root / "eval/models.json").read_text(encoding="utf-8"))
    model = registry["qwen3_4b"]
    if original["model"] != model["model"] or original["revision"] != model["revision"]:
        raise ValueError("Pinned Ukrainian probe model differs")
    scorer = load_generator(model, model["revision"], max_tokens=24)
    outputs = []
    for row in rows:
        prompt = make_prompt(row)
        raw = scorer(prompt)
        codes = parse_codes(raw)
        valid = codes is not None and all(row["source_kind"] in APPLIES_TO[code] for code in codes)
        outputs.append({"id": row["id"], "expected": row["labels"],
                        "predicted": list(codes or ()) if valid else [],
                        "strict_parse": valid, "raw": raw,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    result = {
        "schema_version": 1, "study": "posthoc_consumed_uk_numeral_agreement_only",
        "source_sha256": SOURCE_SHA256, "original_run_sha256": ORIGINAL_SHA256,
        "model": model["model"], "revision": model["revision"],
        "mlx_lm_version": version("mlx-lm"), "max_tokens": 24,
        "changed_rows": changes, "original_a1": _counts(original_rows),
        "corrected_a1": _counts(outputs),
        "decision_changes": [row["id"] for row, old in zip(outputs, original_rows)
                             if row["predicted"] != old["predicted"] or
                             row["strict_parse"] != old["strict_parse"]],
        "outputs": outputs,
        "independent_test": False,
        "ukrainian_language_validated": False,
        "real_child_language_accuracy_established": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    descriptor = os.open(output, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {key: result[key] for key in ("study", "changed_rows", "original_a1",
                                         "corrected_a1", "decision_changes")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose numeral agreement on a consumed Ukrainian probe")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run_diagnostic(args.root.resolve(), args.output), ensure_ascii=False, indent=2))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
