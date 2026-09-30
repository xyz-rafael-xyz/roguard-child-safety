"""Audit one frozen V16 decision across four already consumed D1 tests."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval/runs/ro-v16-d1-transfer-audit.json"
STUDIES = (
    ("batch-0034", "ro-mmbert-v18-test-0034.json", "verify_mmbert_v18.py"),
    ("batch-0036", "ro-mmbert-v19-test-0036.json", "verify_mmbert_v19.py"),
    ("batch-0037", "ro-mmbert-v20-test-0037.json", "verify_mmbert_v20.py"),
    ("batch-0038", "ro-nli-v21-test-0038.json", "verify_nli_v21.py"),
)


def _surface(batch: str, number: int, pairs: list[tuple[float, float]]) -> dict:
    if len(pairs) != 24:
        raise ValueError("Each frozen D1 surface needs 24 negative-to-positive pairs")
    negatives = [left for left, _ in pairs]
    positives = [right for _, right in pairs]
    return {
        "batch": batch, "surface_number": number, "pairs": 24,
        "exact_pairs_at_frozen_cutoff": sum(left < 0.5 <= right for left, right in pairs),
        "positive_recovered": sum(value >= 0.5 for value in positives),
        "negative_false_reviews": sum(value >= 0.5 for value in negatives),
        "positive_above_negative": sum(right > left for left, right in pairs),
        "positive_below_negative": sum(right < left for left, right in pairs),
        "tied_pair_scores": sum(right == left for left, right in pairs),
        "mean_negative_score": round(sum(negatives) / 24, 6),
        "mean_positive_score": round(sum(positives) / 24, 6),
    }


def audit(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifest = root / "models/ro-mmbert-v16-abstract/research.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("shared_threshold") != 0.5:
        raise ValueError("Frozen V16 cutoff differs")
    surfaces, sources = [], []
    for batch, filename, verifier in STUDIES:
        runpy.run_path(str(root / "eval" / verifier))["verify"](root)
        record_path = root / "eval/runs" / filename
        record = json.loads(record_path.read_text(encoding="utf-8"))
        rows = [row for row in load_approved(root, [batch])
                if row["source_kind"] == "message"]
        row_ids = {row["id"] for row in rows}
        saved = [item for item in record["scores"] if item["id"] in row_ids]
        if len(rows) != 96 or len(saved) != 96:
            raise ValueError(f"Expected 96 D1 cards: {batch}")
        scores = []
        for row, item in zip(rows, saved):
            if item["id"] != row["id"] or item["expected"] != row["labels"]:
                raise ValueError(f"Frozen V16 row differs: {row['id']}")
            score = (item["baseline_scores"]["D1"] if batch == "batch-0034"
                     else item["v16_score"])
            if type(score) not in (float, int) or not 0 <= score <= 1:
                raise ValueError(f"Frozen V16 score invalid: {row['id']}")
            scores.append(score)
        for left, right in zip(rows[::2], rows[1::2]):
            if left["labels"] or right["labels"] != ["D1"]:
                raise ValueError(f"D1 reference pair differs: {left['id']}")
        for half in range(2):
            block = scores[half * 48:(half + 1) * 48]
            surfaces.append(_surface(batch, half + 1, list(zip(block[::2], block[1::2]))))
        sources.append({"batch": batch,
                        "batch_sha256": sha256(root / f"data/synthetic/{batch}.jsonl"),
                        "frozen_run_sha256": sha256(record_path),
                        "verifier_sha256": sha256(root / "eval" / verifier)})
    aggregate = {key: sum(surface[key] for surface in surfaces) for key in (
        "pairs", "exact_pairs_at_frozen_cutoff", "positive_recovered",
        "negative_false_reviews", "positive_above_negative",
        "positive_below_negative", "tied_pair_scores")}
    aggregate["worst_surface_exact_pairs"] = min(
        surface["exact_pairs_at_frozen_cutoff"] for surface in surfaces)
    aggregate["best_surface_exact_pairs"] = max(
        surface["exact_pairs_at_frozen_cutoff"] for surface in surfaces)
    return {
        "status": "retrospective_frozen_score_audit_no_model_selection",
        "language": "ro", "category": "D1", "domain": "invented_abstract_cards_only",
        "model": "v16_unchanged", "frozen_cutoff": 0.5,
        "model_manifest_sha256": sha256(manifest),
        "model_weight_sha256": sha256(root / "models/ro-mmbert-v16-abstract/adapter_model.safetensors"),
        "sources": sources, "surfaces": surfaces, "aggregate": aggregate,
        "limitation": "Same-author symbolic surfaces; pooled counts are descriptive and are not independent child-language validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Create the audit once")
    parser.add_argument("--verify", action="store_true", help="Compare with committed audit bytes")
    args = parser.parse_args()
    if args.write == args.verify:
        parser.error("Choose exactly one of --write or --verify")
    rendered = json.dumps(audit(), ensure_ascii=False, indent=2) + "\n"
    if args.write:
        if OUTPUT.exists():
            raise FileExistsError("Preserve the prior transfer audit")
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(OUTPUT)
    elif OUTPUT.read_text(encoding="utf-8") != rendered:
        raise ValueError("Committed V16 transfer audit differs from frozen scores")
    else:
        print("V16 D1 transfer audit matches all frozen scores")


if __name__ == "__main__":
    main()
