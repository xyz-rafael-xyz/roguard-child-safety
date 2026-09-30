"""Post-hoc D1 score-shift diagnostic across frozen, consumed studies."""

from __future__ import annotations

import json
import runpy
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _verify(root: Path, filename: str) -> None:
    verifier = runpy.run_path(str(root / "eval" / filename))["verify"]
    verifier(root)


def _summary(scores: list[float]) -> dict:
    return {"count": len(scores), "median": statistics.median(scores),
            "minimum": min(scores), "maximum": max(scores)}


def analyze(root: Path = ROOT) -> dict:
    for verifier in ("verify_mmbert_v13.py", "verify_mmbert_v14.py",
                     "verify_mmbert_neutral.py"):
        _verify(root, verifier)
    sources = (
        ("abstract_v13", "eval/runs/ro-mmbert-v13-test-0025.json", "predictions"),
        ("abstract_v14", "eval/runs/ro-mmbert-v14-test-0026.json", "model_predictions"),
    )
    result = {}
    thresholds = set()
    for name, path, key in sources:
        run = json.loads((root / path).read_text(encoding="utf-8"))
        thresholds.add(run["threshold"])
        rows = [item for item in run[key] if "D1" in item["scores"]]
        negatives = [item["scores"]["D1"] for item in rows if item["expected"] == []]
        positives = [item["scores"]["D1"] for item in rows if item["expected"] == ["D1"]]
        result[name] = {"negative": _summary(negatives), "positive": _summary(positives)}
    neutral = json.loads((root / "eval/runs/ro-mmbert-neutral-0001.json").read_text(
        encoding="utf-8"))
    thresholds.add(neutral["threshold"])
    result["benign_requests"] = {"negative": _summary(
        [item["score_D1"] for item in neutral["predictions"]])}
    if len(thresholds) != 1:
        raise ValueError("Frozen studies used different D1 cutoffs")
    return {"threshold": thresholds.pop(), "studies": result,
            "interpretation": "Post-hoc score distribution diagnostic; no threshold selection or real-language validity."}


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
