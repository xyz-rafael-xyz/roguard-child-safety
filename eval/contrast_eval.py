"""Evaluate a frozen one-factor set without using it for training or tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.metrics import metrics
from roguard.model import MLXClassifier
from roguard.prompt import make_prompt_v2, parse_codes
from roguard.review import APPLIES_TO, sha256
from roguard.screen import screen

from contrast_suite import build_rows
from run_panel import load_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", choices=("ro-v2", "romistral-base", "qwen3-base"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data_path = root / "data" / "synthetic" / "contrast-0001.jsonl"
    manifest = json.loads(data_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    if rows != build_rows() or manifest["sha256"] != sha256(data_path) or manifest["training_eligible"]:
        raise ValueError("Contrast set differs from the frozen symbolic generator")
    if args.model == "ro-v2":
        adapter = root / "checkpoints" / "ro-v2" / "adapter"
        metadata = json.loads((adapter / "roguard_metadata.json").read_text(encoding="utf-8"))
        if metadata["prompt_version"] != "v2" or metadata["language"] != "ro":
            raise ValueError("Adapter metadata differs from fixed contrast protocol")
        backend = MLXClassifier(adapter)
        thresholds = {code: 0.5 for code in ("D1", "R1", "A1", "P1", "G1", "S1")}

        def predict(row: dict) -> tuple[tuple[str, ...], bool, str | None]:
            try:
                result = screen(row["text"], language="ro", source_kind=row["source_kind"],
                                backend=backend, thresholds=thresholds)
                return result.labels, True, None
            except ValueError as exc:
                if "Model output could not be parsed" not in str(exc):
                    raise
                return (), False, None

        revision = metadata["base_revision"]
    else:
        registry = json.loads((root / "eval" / "models.json").read_text(encoding="utf-8"))
        entry = registry["romistral7b" if args.model == "romistral-base" else "qwen3_4b"]
        revision = entry["revision"]
        run = load_generator(entry, revision, max_tokens=24)

        def predict(row: dict) -> tuple[tuple[str, ...], bool, str | None]:
            raw = run(make_prompt_v2(row))
            parsed = parse_codes(raw)
            valid = parsed is not None and all(row["source_kind"] in APPLIES_TO[code] for code in parsed)
            return tuple(parsed or ()), valid, raw

    outputs = []
    predictions = {}
    for row in rows:
        codes, valid, raw = predict(row)
        if not valid:
            codes = ()
        predictions[row["id"]] = codes
        outputs.append({"id": row["id"], "pair_id": row["pair_id"], "variant": row["variant"],
                        "expected": row["labels"], "predicted": codes, "strict_parse": valid,
                        "raw": raw})
    exact_rows = sum(tuple(row["labels"]) == predictions[row["id"]] for row in rows)
    exact_pairs = 0
    correct_flips = 0
    for negative, positive in zip(rows[::2], rows[1::2]):
        code = positive["labels"][0]
        neg_codes, pos_codes = predictions[negative["id"]], predictions[positive["id"]]
        exact_pairs += neg_codes == () and pos_codes == (code,)
        correct_flips += code not in neg_codes and code in pos_codes
    result = {
        "name": "contrast-0001", "model": args.model, "revision": revision,
        "prompt_version": "v2", "max_tokens": 24,
        "data_sha256": sha256(data_path), "rows": len(rows), "pairs": len(rows) // 2,
        "parsed_rows": sum(item["strict_parse"] for item in outputs),
        "exact_rows": exact_rows, "exact_pairs": exact_pairs,
        "correct_direction_flips": correct_flips,
        "metrics": metrics(rows, predictions), "outputs": outputs,
        "limitation": "Symbolic one-factor descriptions, not child language; the same decision rules are visible in the taxonomy.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Contrast result already exists")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
