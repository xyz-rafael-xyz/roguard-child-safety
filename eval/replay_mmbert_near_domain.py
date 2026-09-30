"""Replay every saved near-domain score with the pinned local base and adapters."""

from __future__ import annotations

import argparse
import gc
import json
import runpy
from pathlib import Path

from roguard import D1FactorResearchClassifier, MMBertResearchClassifier
from roguard.prompt_v18 import FIELDS, decision

ROOT = Path(__file__).resolve().parents[1]


def replay(base_model_path: Path, tolerance: float = 1e-5) -> dict:
    verifier = runpy.run_path(str(ROOT / "eval/verify_mmbert_near_domain.py"))
    verifier["verify"](ROOT)
    runner = runpy.run_path(str(ROOT / "eval/run_mmbert_near_domain.py"))
    record = json.loads((ROOT / runner["OUTPUT"]).read_text(encoding="utf-8"))
    samples = runner["rows"](ROOT)
    result = {}
    for name, directory in runner["MODELS"].items():
        backend = (D1FactorResearchClassifier(base_model_path, ROOT / directory)
                   if name == "v18" else MMBertResearchClassifier(base_model_path, ROOT / directory))
        expected = record["results"][name]
        max_delta = 0.0
        for (group, sample), saved in zip(samples, expected["predictions"]):
            if saved["group"] != group:
                raise ValueError(f"Saved group changed for {name}")
            text = runner["PREFIX"] + sample
            scores = (backend.score(text) if name == "v18" else
                      {"D1": backend.score(text, "ro", "message")["D1"]})
            for field, value in scores.items():
                delta = abs(value - saved["scores"][field])
                max_delta = max(max_delta, delta)
                if delta > tolerance:
                    raise ValueError(f"Saved model score differs: {name}, {field}, {saved['id']}")
            cutoff = expected["threshold"]
            predicted = (decision({field: scores[field] >= cutoff for field in FIELDS})
                         if name == "v18" else scores["D1"] >= cutoff)
            if predicted != saved["predicted"]:
                raise ValueError(f"Saved model decision differs: {name}, {saved['id']}")
        result[name] = {"replayed": len(samples), "max_absolute_score_delta": max_delta,
                        "all_decisions_match": True}
        del backend
        gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.base_model_path), indent=2))


if __name__ == "__main__":
    main()
