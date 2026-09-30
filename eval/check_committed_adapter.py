"""Load the private research adapter and reproduce one saved symbolic score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard import MMBertResearchClassifier, screen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    adapter = ROOT / "models/ro-mmbert-v11-abstract"
    backend = MMBertResearchClassifier(args.base_model_path, adapter)
    row = json.loads((ROOT / "data/synthetic/batch-0024.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    prior = json.loads((ROOT / "eval/runs/ro-mmbert-v12-test-0024.json").read_text(
        encoding="utf-8"))["v11_predictions"][0]
    result = screen(row["text"], language="ro", source_kind=row["source_kind"],
                    backend=backend, thresholds=backend.thresholds)
    score_delta = abs(result.scores["D1"] - prior["scores"]["D1"])
    matched = score_delta < 1e-5 and list(result.labels) == prior["predicted"]
    print(json.dumps({"saved_score_match": matched, "absolute_score_delta": score_delta,
                      "scope": "one_consumed_symbolic_card_reproduction"}, indent=2))
    if not matched:
        raise SystemExit("Committed adapter inference differs from the frozen run")


if __name__ == "__main__":
    main()
