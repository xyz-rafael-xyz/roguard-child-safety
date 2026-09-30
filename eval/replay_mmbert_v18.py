"""Replay the failed V18 test from pinned weights without changing its record."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from roguard import D1FactorResearchClassifier, MMBertResearchClassifier
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v18-test-0034.json")
BASELINE = Path("models/ro-mmbert-v16-abstract")
FACTOR = Path("models/ro-mmbert-v18-factor-abstract")


def replay(base_model_path: Path, tolerance: float = 1e-5) -> dict:
    checked = runpy.run_path(str(ROOT / "eval/verify_mmbert_v18.py"))["verify"](ROOT)
    record = json.loads((ROOT / RUN).read_text(encoding="utf-8"))
    rows = load_approved(ROOT, ["batch-0034"])
    baseline = MMBertResearchClassifier(base_model_path, ROOT / BASELINE)
    factor = D1FactorResearchClassifier(base_model_path, ROOT / FACTOR)
    max_delta = 0.0
    field_count = 0
    for row, saved in zip(rows, record["scores"]):
        if row["id"] != saved["id"]:
            raise ValueError("V18 saved test row order differs")
        scores = baseline.score(row["text"], "ro", row["source_kind"])
        for code in applicable_codes(row["source_kind"]):
            delta = abs(scores[code] - saved["baseline_scores"][code])
            max_delta = max(max_delta, delta)
            field_count += 1
            if delta > tolerance:
                raise ValueError(f"V16 test score changed: {row['id']} {code}")
        if row["source_kind"] == "message":
            scores = factor.score(row["text"])
            for name, value in scores.items():
                delta = abs(value - saved["factor_scores"][name])
                max_delta = max(max_delta, delta)
                field_count += 1
                if delta > tolerance:
                    raise ValueError(f"V18 factor test score changed: {row['id']} {name}")
    return {"cards_replayed": len(rows), "scores_replayed": field_count,
            "max_absolute_score_delta": max_delta,
            "candidate_pairs": checked["v18_pairs"],
            "candidate_registered_target_met": checked["preregistered_success"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.base_model_path), indent=2))


if __name__ == "__main__":
    main()
