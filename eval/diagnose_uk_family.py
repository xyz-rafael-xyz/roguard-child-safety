"""Run preregistered prompt-only model-family diagnostics on a consumed UK probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from roguard.prompt import parse_codes
from roguard.review import APPLIES_TO, sha256, taxonomy_sha256

from diagnose_uk_numerals import ROOT, SOURCE_SHA256, corrected_rows
from run_panel import load_generator
from uk_contrast_suite import build_rows
from uk_prompt_probe import make_prompt

REGISTRY_SHA256 = "bc089718f74389a252eec9cd30d4a9ecbc7dd136b482f3536239e52a3775ac79"
PROMPT_SHA256 = "d35ea6cccd404ad49b5c011d7b4faddbdeb66cc78be71d80cfae34f7be6def50"
CORRECTION_SHA256 = "9f12b6415acca383fe667e73d88bc70e9a4f9a1d15ba8556649ec93a1e3c4e9d"
MODELS = ("phi4_mini", "gemma3_4b", "mistral7b")


def corrected_probe(root: Path) -> list[dict]:
    source = root / "data/synthetic/uk-contrast-0001.jsonl"
    if (sha256(source) != SOURCE_SHA256 or
            sha256(root / "eval/runs/uk-numeral-agreement-diagnostic.json") != CORRECTION_SHA256):
        raise ValueError("Frozen Ukrainian diagnostic bytes differ")
    source_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if source_rows != build_rows():
        raise ValueError("Frozen Ukrainian generator differs")
    a1_rows, _ = corrected_rows(root)
    corrected = {row["id"]: row for row in a1_rows}
    rows = [corrected.get(row["id"], row) for row in source_rows]
    if len(rows) != 48 or sum(row["text"] != prior["text"] for row, prior in zip(rows, source_rows)) != 5:
        raise ValueError("Corrected Ukrainian probe differs")
    return rows


def category_counts(rows: list[dict], outputs: list[dict]) -> list[dict]:
    """Keep parse abstentions outside the four confusion cells."""
    result = []
    for code in APPLIES_TO:
        counts = {name: 0 for name in (
            "tp", "fp", "fn", "tn", "positive_abstentions", "negative_abstentions")}
        for row, output in zip(rows, outputs):
            if row["source_kind"] not in APPLIES_TO[code]:
                continue
            expected = code in row["labels"]
            if not output["strict_parse"]:
                counts["positive_abstentions" if expected else "negative_abstentions"] += 1
                continue
            predicted = code in output["predicted"]
            counts["tp" if predicted and expected else
                   "fp" if predicted else "fn" if expected else "tn"] += 1
        positive = counts["tp"] + counts["fn"] + counts["positive_abstentions"]
        negative = counts["tn"] + counts["fp"] + counts["negative_abstentions"]
        result.append({"category": code, "support": positive, "negative_support": negative,
                       **counts,
                       "precision": counts["tp"] / (counts["tp"] + counts["fp"])
                       if counts["tp"] + counts["fp"] else None,
                       "recall_with_abstentions_as_misses": counts["tp"] / positive if positive else None,
                       "specificity_with_abstentions_as_misses": counts["tn"] / negative if negative else None})
    return result


def run(root: Path, model_name: str, output: Path) -> dict:
    if model_name not in MODELS:
        raise ValueError("Unregistered Ukrainian diagnostic model")
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Diagnostic output already exists")
    if (sha256(root / "eval/models.json") != REGISTRY_SHA256 or
            sha256(root / "eval/uk_prompt_probe.py") != PROMPT_SHA256):
        raise ValueError("Frozen model registry or Ukrainian prompt differs")
    rows = corrected_probe(root)
    model = json.loads((root / "eval/models.json").read_text(encoding="utf-8"))[model_name]
    if model["engine"] != "mlx":
        raise ValueError("Expected pinned MLX generator")
    scorer = load_generator(model, model["revision"], max_tokens=24)
    outputs = []
    for row in rows:
        prompt = make_prompt(row)
        raw = scorer(prompt)
        codes = parse_codes(raw)
        valid = codes is not None and all(row["source_kind"] in APPLIES_TO[code] for code in codes)
        predicted = list(codes or ()) if valid else []
        outputs.append({"id": row["id"], "pair_id": row["pair_id"], "variant": row["variant"],
                        "expected": row["labels"], "predicted": predicted,
                        "strict_parse": valid, "raw": raw,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    exact_pairs = sum(left["strict_parse"] and right["strict_parse"] and
                      left["predicted"] == [] and right["predicted"] == right["expected"]
                      for left, right in zip(outputs[::2], outputs[1::2]))
    flips = sum(left["strict_parse"] and right["strict_parse"] and
                right["expected"][0] not in left["predicted"] and
                right["expected"][0] in right["predicted"]
                for left, right in zip(outputs[::2], outputs[1::2]))
    result = {
        "schema_version": 1, "study": "posthoc_consumed_uk_model_family_only",
        "independent_test": False, "ukrainian_language_validated": False,
        "real_child_language_accuracy_established": False,
        "source_sha256": SOURCE_SHA256, "corrected_a1_diagnostic_sha256": CORRECTION_SHA256,
        "registry_sha256": REGISTRY_SHA256, "prompt_code_sha256": PROMPT_SHA256,
        "taxonomy_sha256": taxonomy_sha256(root, "uk"),
        "model_key": model_name, "model": model["model"], "revision": model["revision"],
        "mlx_lm_version": version("mlx-lm"), "max_tokens": 24,
        "rows": len(rows), "pairs": len(rows) // 2,
        "parsed_rows": sum(item["strict_parse"] for item in outputs),
        "exact_rows": sum(item["strict_parse"] and item["predicted"] == item["expected"]
                          for item in outputs),
        "exact_pairs": exact_pairs, "correct_direction_flips": flips,
        "category_counts": category_counts(rows, outputs), "outputs": outputs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    descriptor = os.open(output, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {key: result[key] for key in ("model_key", "parsed_rows", "exact_rows",
                                         "exact_pairs", "correct_direction_flips")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a preregistered consumed Ukrainian diagnostic")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.root.resolve(), args.model, args.output), indent=2))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
