"""Retrospective D1 fusion diagnostic on consumed development cards only."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval/runs/ro-v22-ensemble-development-audit.json"
BASELINE = {
    "batch-0037": "ro-mmbert-v20-test-0037.json",
    "batch-0038": "ro-nli-v21-test-0038.json",
}


def audit(root: Path = ROOT) -> dict:
    root = root.resolve()
    verified = runpy.run_path(str(root / "eval/verify_committed_v22_selection.py"))["verify"](root)
    selected = root / f"eval/runs/ro-mmbert-v22-epoch-{verified['selected_epoch']}-dev.json"
    v22 = json.loads(selected.read_text(encoding="utf-8"))["predictions"]
    grouped = {name: [] for name in ("v16", "v22", "mean")}
    v22_subset_of_v16 = True
    for batch, name in BASELINE.items():
        frozen = json.loads((root / "eval/runs" / name).read_text(encoding="utf-8"))["scores"]
        rows = v22[batch]
        for half in (0, 1):
            offsets = range(half * 48, (half + 1) * 48, 2)
            pairs = []
            for index in offsets:
                left, right = frozen[index:index + 2]
                newer_left, newer_right = rows[index:index + 2]
                if (left["id"] != newer_left["id"] or right["id"] != newer_right["id"] or
                        left["expected"] != newer_left["expected"] or
                        right["expected"] != newer_right["expected"] or
                        left["expected"] != [] or right["expected"] != ["D1"]):
                    raise ValueError("Frozen development pair alignment differs")
                old = [left["v16_score"], right["v16_score"]]
                new = [newer_left["scores"]["D1"], newer_right["scores"]["D1"]]
                if any(b >= 0.5 and a < 0.5 for a, b in zip(old, new)):
                    v22_subset_of_v16 = False
                pairs.append({"v16": old, "v22": new,
                              "mean": [(a + b) / 2 for a, b in zip(old, new)]})
            for method in grouped:
                values = [pair[method] for pair in pairs]
                grouped[method].append({
                    "batch": batch, "surface": half + 1,
                    "exact_pairs": sum(left < 0.5 <= right for left, right in values),
                    "positive_recovered": sum(right >= 0.5 for _, right in values),
                    "false_reviews": sum(left >= 0.5 for left, _ in values),
                    "positive_above_negative": sum(right > left for left, right in values),
                })
    summary = {
        method: {
            "surface_pairs": [surface["exact_pairs"] for surface in surfaces],
            "exact_pairs": sum(surface["exact_pairs"] for surface in surfaces),
            "positive_recovered": sum(surface["positive_recovered"] for surface in surfaces),
            "false_reviews": sum(surface["false_reviews"] for surface in surfaces),
            "positive_above_negative": sum(surface["positive_above_negative"] for surface in surfaces),
        }
        for method, surfaces in grouped.items()
    }
    if summary["v16"] != {
            "surface_pairs": [10, 23, 18, 24], "exact_pairs": 75,
            "positive_recovered": 85, "false_reviews": 13,
            "positive_above_negative": 89}:
        raise ValueError("Frozen V16 development comparator differs")
    return {
        "status": "retrospective_consumed_development_diagnostic_no_selection",
        "domain": "same_author_invented_romanian_abstract_cards",
        "selected_v22_epoch": verified["selected_epoch"],
        "v22_development_gate_passed": verified["development_gate_passed"],
        "v22_review_decisions_subset_of_v16": v22_subset_of_v16,
        "source_sha256": {batch: sha256(root / "eval/runs" / filename)
                          for batch, filename in BASELINE.items()} | {
            "v22": sha256(selected)},
        "summary": summary,
        "limitation": "Consumed, correlated abstract cards; no candidate selected or new held-out accuracy estimated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write == args.verify:
        parser.error("Choose exactly one of --write or --verify")
    rendered = json.dumps(audit(), ensure_ascii=False, indent=2) + "\n"
    if args.write:
        if OUTPUT.exists():
            raise FileExistsError("Preserve the original development audit")
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(OUTPUT)
    elif OUTPUT.read_text(encoding="utf-8") != rendered:
        raise ValueError("V22 fusion diagnostic differs from frozen development scores")
    else:
        print("V22 fusion diagnostic matches frozen development scores")


if __name__ == "__main__":
    main()
