"""Evaluate pinned prompt-only models with the frozen Romanian binary prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from roguard.prompt_v4 import applicable_codes, make_prompt_v4, parse_binary
from roguard.qwen_output import PARSER_PROFILE, parse_qwen_binary
from roguard.review import ReviewError, load_approved, taxonomy_sha256

from run_panel import load_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", required=True)
    parser.add_argument("--model", required=True, choices=("qwen3_4b", "phi4_mini", "gemma3_4b", "mistral7b", "romistral7b"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-empty-think-wrapper", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_approved(root, [args.batch])
    if not rows or any(row["split"] != "test" for row in rows):
        raise ReviewError("Binary panel accepts held-out test cards only")
    entry = json.loads((root / "eval" / "models.json").read_text(encoding="utf-8"))[args.model]
    if args.qwen_empty_think_wrapper and args.model != "qwen3_4b":
        raise ValueError("Qwen wrapper parser requires the pinned Qwen model")
    output_parser = parse_qwen_binary if args.qwen_empty_think_wrapper else parse_binary
    run = load_generator(entry, entry["revision"], max_tokens=8)
    outputs = []
    for row in rows:
        started = time.monotonic()
        raw = {}
        digests = {}
        predicted = []
        valid = True
        for code in applicable_codes(row["source_kind"]):
            prompt = make_prompt_v4(row, code)
            digests[code] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            raw[code] = run(prompt)
            answer = output_parser(raw[code])
            if answer is None:
                valid = False
            elif answer:
                predicted.append(code)
        if not valid:
            predicted = []
        outputs.append({"id": row["id"], "expected": row["labels"], "predicted": predicted,
                        "strict_parse": valid, "raw": raw, "prompt_sha256": digests,
                        "seconds": round(time.monotonic() - started, 3)})
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Do not overwrite a panel run")
    output.write_text(json.dumps({"model": entry["model"], "revision": entry["revision"],
                                  "source_model": entry.get("source_model", entry["model"]),
                                  "engine": entry["engine"], "prompt_version": "v4_binary",
                                  "max_tokens_per_task": 8, "batches": [args.batch],
                                  "parser_profile": PARSER_PROFILE if args.qwen_empty_think_wrapper else "literal_da_nu_v1",
                                  "taxonomy_sha256": {"ro": taxonomy_sha256(root, "ro")},
                                  "outputs": outputs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
