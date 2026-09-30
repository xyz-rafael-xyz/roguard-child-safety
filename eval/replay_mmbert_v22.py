"""Replay the selected V22 adapter on its two consumed development batches."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from roguard import MMBertResearchClassifier
from roguard.review import load_approved

ROOT = Path(__file__).resolve().parents[1]


def replay(base_model_path: Path, tolerance: float = 1e-5) -> dict:
    checked = runpy.run_path(str(ROOT / "eval/verify_committed_v22_selection.py"))["verify"](ROOT)
    model = MMBertResearchClassifier(base_model_path, ROOT / "models/ro-mmbert-v22-abstract")
    epoch = checked["selected_epoch"]
    saved = json.loads((ROOT / f"eval/runs/ro-mmbert-v22-epoch-{epoch}-dev.json").read_text())
    maximum, count = 0.0, 0
    for batch in ("batch-0037", "batch-0038"):
        rows = [row for row in load_approved(ROOT, [batch]) if row["source_kind"] == "message"]
        items = saved["predictions"][batch]
        if len(rows) != 96 or len(items) != len(rows):
            raise ValueError("V22 development replay count differs")
        for row, item in zip(rows, items):
            if row["id"] != item["id"]:
                raise ValueError("V22 development replay row differs")
            score = model.score(row["text"], "ro", "message")["D1"]
            delta = abs(score - item["scores"]["D1"])
            maximum = max(maximum, delta)
            count += 1
            if delta > tolerance:
                raise ValueError(f"V22 score differs: {row['id']}")
    return {"cards_replayed": count, "max_absolute_score_delta": maximum,
            "selected_epoch": epoch,
            "development_gate_passed": checked["development_gate_passed"],
            "new_held_out_test_exists": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.base_model_path), indent=2))


if __name__ == "__main__":
    main()
