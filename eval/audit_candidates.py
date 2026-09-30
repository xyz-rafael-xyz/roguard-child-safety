"""Read-only leakage and balance audit for explicitly named synthetic candidates."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

from roguard.review import candidate_manifest


def audit(root: Path, batch_ids: list[str]) -> dict:
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("Specify each batch once")
    rows_by_batch = {}
    hashes = {}
    for batch_id in batch_ids:
        hashes[batch_id] = candidate_manifest(root, batch_id)["batch_sha256"]
        path = root / "data" / "synthetic" / f"{batch_id}.jsonl"
        rows_by_batch[batch_id] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    all_rows = [row for rows in rows_by_batch.values() for row in rows]
    texts = [(row["language"], row["source_kind"], " ".join(row["text"].casefold().split())) for row in all_rows]
    leaks = [row["id"] for row in all_rows if re.search(r"\b(?:D1|R1|A1|P1|G1|S1)\b", row["text"])]
    comparisons = []
    for source_id, source_rows in rows_by_batch.items():
        for target_id, target_rows in rows_by_batch.items():
            if source_id >= target_id:
                continue
            ratios = [max((difflib.SequenceMatcher(None, row["text"], other["text"]).ratio()
                           for other in source_rows if other["source_kind"] == row["source_kind"]), default=0)
                      for row in target_rows]
            comparisons.append({"source": source_id, "target": target_id,
                                "mean_nearest_similarity": round(sum(ratios) / len(ratios), 3),
                                "max_nearest_similarity": round(max(ratios), 3),
                                "rows_above_0_90": sum(value > 0.9 for value in ratios)})
    return {"batch_sha256": hashes,
            "rows": {batch_id: len(rows) for batch_id, rows in rows_by_batch.items()},
            "positive_labels": {batch_id: dict(Counter(code for row in rows for code in row["labels"]))
                                for batch_id, rows in rows_by_batch.items()},
            "exact_duplicate_texts": len(texts) - len(set(texts)),
            "category_codes_in_card_text": leaks,
            "character_similarity_by_source_kind": comparisons,
            "note": "Similarity is a superficial leakage check; it does not establish semantic independence."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.batch)
    if args.output.exists():
        raise FileExistsError("Do not overwrite an audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
