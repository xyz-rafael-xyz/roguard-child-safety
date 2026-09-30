"""Retrospective bound on what one V16 cutoff could achieve on consumed cards."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "eval/runs/ro-v16-cutoff-feasibility.json"
INPUT = ROOT / "eval/runs/ro-v16-d1-transfer-audit.json"


def audit() -> dict:
    prior = runpy.run_path(str(ROOT / "eval/audit_v16_d1_transfer.py"))
    expected = json.dumps(prior["audit"](), ensure_ascii=False, indent=2) + "\n"
    if INPUT.read_text(encoding="utf-8") != expected:
        raise ValueError("Frozen V16 cross-batch audit differs")
    surfaces = []
    for batch, filename, _ in prior["STUDIES"]:
        record = json.loads((ROOT / "eval/runs" / filename).read_text(encoding="utf-8"))
        scores = [item["baseline_scores"]["D1"] if batch == "batch-0034"
                  else item["v16_score"] for item in record["scores"][:96]]
        if len(scores) != 96:
            raise ValueError(f"Frozen D1 score count differs: {batch}")
        for half in range(2):
            block = scores[half * 48:(half + 1) * 48]
            surfaces.append(list(zip(block[::2], block[1::2])))
    values = sorted({score for pairs in surfaces for pair in pairs for score in pair})
    thresholds = sorted({0.0, 1.0, *values,
                         *((left + right) / 2 for left, right in zip(values, values[1:]))})
    counts = [tuple(sum(left < cutoff <= right for left, right in pairs)
                    for pairs in surfaces) for cutoff in thresholds]
    frozen_counts = tuple(surface["exact_pairs_at_frozen_cutoff"]
                          for surface in json.loads(expected)["surfaces"])
    if tuple(sum(left < 0.5 <= right for left, right in pairs)
             for pairs in surfaces) != frozen_counts:
        raise ValueError("Frozen V16 cutoff counts differ")
    return {
        "status": "retrospective_oracle_bound_not_a_selected_cutoff",
        "source_audit_sha256": sha256(INPUT),
        "scope": "all_eight_same_author_consumed_abstract_D1_surfaces",
        "candidate_cutoffs": len(thresholds),
        "frozen_cutoff": 0.5,
        "frozen_complete_pairs": sum(frozen_counts),
        "frozen_worst_surface_pairs": min(frozen_counts),
        "posthoc_maximum_complete_pairs_under_one_cutoff": max(map(sum, counts)),
        "posthoc_maximum_worst_surface_pairs_under_one_cutoff": max(map(min, counts)),
        "posthoc_per_surface_maximum_pairs": [max(item[index] for item in counts)
                                                for index in range(len(surfaces))],
        "limitation": "Searches consumed test labels solely to diagnose feasibility; no cutoff from this search may be selected or reported as held-out performance.",
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
            raise FileExistsError("Preserve the prior cutoff feasibility audit")
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(OUTPUT)
    elif OUTPUT.read_text(encoding="utf-8") != rendered:
        raise ValueError("Committed cutoff feasibility audit differs")
    else:
        print("V16 cutoff feasibility matches the consumed scores")


if __name__ == "__main__":
    main()
