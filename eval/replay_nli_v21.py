"""Replay every saved V21 and comparator test score from frozen adapters."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from roguard import (BalancedJointD1ResearchClassifier, D1FactorResearchClassifier,
                     MMBertResearchClassifier, NLIResearchClassifier)
from roguard.review import load_approved

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-nli-v21-test-0038.json")


def replay(nli_base_model_path: Path, mmbert_base_model_path: Path,
           tolerance: float = 1e-5) -> dict:
    checked = runpy.run_path(str(ROOT / "eval/verify_nli_v21.py"))["verify"](ROOT)
    record = json.loads((ROOT / RUN).read_text(encoding="utf-8"))
    rows = load_approved(ROOT, ["batch-0038"])
    models = {
        "v21": NLIResearchClassifier(nli_base_model_path, ROOT / "models/ro-nli-v21-abstract"),
        "v20": BalancedJointD1ResearchClassifier(
            mmbert_base_model_path, ROOT / "models/ro-mmbert-v20-balanced-joint-abstract"),
        "v18": D1FactorResearchClassifier(
            mmbert_base_model_path, ROOT / "models/ro-mmbert-v18-factor-abstract"),
        "v16": MMBertResearchClassifier(
            mmbert_base_model_path, ROOT / "models/ro-mmbert-v16-abstract"),
    }
    maximum, count = 0.0, 0
    for row, saved in zip(rows, record["scores"]):
        for name in ("v21", "v20", "v18"):
            current = models[name].score(row["text"])
            for field, value in current.items():
                delta = abs(value - saved[f"{name}_scores"][field])
                maximum = max(maximum, delta)
                count += 1
                if delta > tolerance:
                    raise ValueError(f"Frozen {name} score differs: {row['id']} {field}")
        value = models["v16"].score(row["text"], "ro", "message")["D1"]
        delta = abs(value - saved["v16_score"])
        maximum = max(maximum, delta)
        count += 1
        if delta > tolerance:
            raise ValueError(f"Frozen V16 score differs: {row['id']}")
    return {"cards_replayed": len(rows), "scores_replayed": count,
            "max_absolute_score_delta": maximum, "pairs": checked["pairs"],
            "candidate_registered_target_met": checked["preregistered_success"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nli-base-model-path", type=Path, required=True)
    parser.add_argument("--mmbert-base-model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.nli_base_model_path, args.mmbert_base_model_path), indent=2))


if __name__ == "__main__":
    main()
