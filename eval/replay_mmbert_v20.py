"""Replay all V20 test scores with the pinned local base and four adapters."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from roguard import (BalancedJointD1ResearchClassifier, D1FactorResearchClassifier,
                     JointD1ResearchClassifier, MMBertResearchClassifier)
from roguard.review import load_approved

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v20-test-0037.json")


def replay(base_model_path: Path, tolerance: float = 1e-5) -> dict:
    checked = runpy.run_path(str(ROOT / "eval/verify_mmbert_v20.py"))["verify"](ROOT)
    record = json.loads((ROOT / RUN).read_text(encoding="utf-8"))
    rows = load_approved(ROOT, ["batch-0037"])
    models = {
        "v20": BalancedJointD1ResearchClassifier(
            base_model_path, ROOT / "models/ro-mmbert-v20-balanced-joint-abstract"),
        "v19": JointD1ResearchClassifier(
            base_model_path, ROOT / "models/ro-mmbert-v19-joint-abstract"),
        "v18": D1FactorResearchClassifier(
            base_model_path, ROOT / "models/ro-mmbert-v18-factor-abstract"),
        "v16": MMBertResearchClassifier(
            base_model_path, ROOT / "models/ro-mmbert-v16-abstract"),
    }
    max_delta = 0.0
    count = 0
    for row, saved in zip(rows, record["scores"]):
        for name in ("v20", "v19", "v18"):
            current = models[name].score(row["text"])
            for field, value in current.items():
                delta = abs(value - saved[f"{name}_scores"][field])
                max_delta = max(max_delta, delta)
                count += 1
                if delta > tolerance:
                    raise ValueError(f"Frozen {name} score differs: {row['id']} {field}")
        value = models["v16"].score(row["text"], "ro", "message")["D1"]
        delta = abs(value - saved["v16_score"])
        max_delta = max(max_delta, delta)
        count += 1
        if delta > tolerance:
            raise ValueError(f"Frozen V16 score differs: {row['id']}")
    return {"cards_replayed": len(rows), "scores_replayed": count,
            "max_absolute_score_delta": max_delta, "pairs": checked["pairs"],
            "candidate_registered_target_met": checked["preregistered_success"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.base_model_path), indent=2))


if __name__ == "__main__":
    main()
