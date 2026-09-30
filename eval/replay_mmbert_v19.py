"""Replay every frozen V19, V18 and V16 test score from pinned local weights."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from roguard import (D1FactorResearchClassifier, JointD1ResearchClassifier,
                     MMBertResearchClassifier)
from roguard.review import load_approved

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v19-test-0036.json")


def replay(base_model_path: Path, tolerance: float = 1e-5) -> dict:
    checked = runpy.run_path(str(ROOT / "eval/verify_mmbert_v19.py"))["verify"](ROOT)
    record = json.loads((ROOT / RUN).read_text(encoding="utf-8"))
    rows = load_approved(ROOT, ["batch-0036"])
    joint = JointD1ResearchClassifier(base_model_path, ROOT / "models/ro-mmbert-v19-joint-abstract")
    factor = D1FactorResearchClassifier(base_model_path, ROOT / "models/ro-mmbert-v18-factor-abstract")
    direct = MMBertResearchClassifier(base_model_path, ROOT / "models/ro-mmbert-v16-abstract")
    max_delta = 0.0
    count = 0
    for row, saved in zip(rows, record["scores"]):
        current = {
            "joint_scores": joint.score(row["text"]),
            "v18_scores": factor.score(row["text"]),
            "v16_score": direct.score(row["text"], "ro", "message")["D1"],
        }
        for name in ("joint_scores", "v18_scores"):
            for field, value in current[name].items():
                delta = abs(value - saved[name][field])
                max_delta = max(max_delta, delta)
                count += 1
                if delta > tolerance:
                    raise ValueError(f"Frozen model score differs: {row['id']} {name} {field}")
        delta = abs(current["v16_score"] - saved["v16_score"])
        max_delta = max(max_delta, delta)
        count += 1
        if delta > tolerance:
            raise ValueError(f"Frozen V16 score differs: {row['id']}")
    return {"cards_replayed": len(rows), "scores_replayed": count,
            "max_absolute_score_delta": max_delta,
            "pairs": checked["pairs"],
            "candidate_registered_target_met": checked["preregistered_success"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.base_model_path), indent=2))


if __name__ == "__main__":
    main()
