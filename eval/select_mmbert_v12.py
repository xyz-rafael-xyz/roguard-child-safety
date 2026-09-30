"""Select per-category cutoffs on attested development rows only."""

from __future__ import annotations

import itertools
import json
import math
import subprocess
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.review import CATEGORIES, load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
DEV_SCORES = Path("eval/runs/ro-mmbert-v11-epoch-3-dev.json")
V11_CHOICE = Path("eval/runs/ro-mmbert-v11-dev-selection.json")
OUTPUT = Path("eval/runs/ro-mmbert-v12-dev-selection.json")
GROUPS = (("D1",), ("R1",), ("P1",), ("G1",), ("A1", "S1"))


def apply_thresholds(items: list[dict], thresholds: dict[str, float]) -> list[dict]:
    if set(thresholds) != set(CATEGORIES) or any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds.values()
    ):
        raise ValueError("All six category thresholds must be finite and within [0, 1]")
    return [{**item, "predicted": [code for code in CATEGORIES if code in item["scores"]
                                     and item["scores"][code] >= thresholds[code]]} for item in items]


def candidates(items: list[dict], code: str) -> list[float]:
    scores = sorted({item["scores"][code] for item in items if code in item["scores"]})
    if not scores or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError(f"Invalid development scores for {code}")
    return sorted({0.0, 0.5, 1.0} | {(left + right) / 2 for left, right in zip(scores, scores[1:])})


def select_thresholds(rows: list[dict], items: list[dict]) -> tuple[dict[str, float], dict, dict[str, int]]:
    if len(rows) != 144 or len(items) != 144:
        raise ValueError("V12 requires all 144 attested development rows")
    evaluate_pairs(rows, [{**item, "predicted": []} for item in items])
    chosen: dict[str, float] = {}
    counts = {}
    for group in GROUPS:
        group_ids = {item["id"] for item in items if any(code in item["scores"] for code in group)}
        group_rows = [row for row in rows if row["id"] in group_ids]
        group_items = [item for item in items if item["id"] in group_ids]
        grids = [candidates(group_items, code) for code in group]
        counts["+".join(group)] = math.prod(len(grid) for grid in grids)
        best = None
        for values in itertools.product(*grids):
            options = dict(zip(group, values))
            predictions = [{**item, "predicted": [code for code in group
                                                  if code in item["scores"] and
                                                  item["scores"][code] >= options[code]]}
                           for item in group_items]
            report = evaluate_pairs(group_rows, predictions)
            key = (report["exact_pairs"], report["exact_cards"],
                   -report["false_review_on_negatives"],
                   -sum(abs(value - 0.5) for value in values),
                   tuple(-value for value in values))
            if best is None or key > best[0]:
                best = (key, options)
        assert best is not None
        chosen.update(best[1])
    report = evaluate_pairs(rows, apply_thresholds(items, chosen))
    return chosen, report, counts


def validated_inputs(root: Path) -> tuple[list[dict], list[dict], dict]:
    choice = json.loads((root / V11_CHOICE).read_text(encoding="utf-8"))
    dev = json.loads((root / DEV_SCORES).read_text(encoding="utf-8"))
    selected = next(candidate for candidate in choice["candidates"]
                    if candidate["epoch"] == choice["selected_epoch"])
    if (choice["selected_epoch"] != 3 or
            selected["dev_predictions_sha256"] != sha256(root / DEV_SCORES) or
            dev["batches"] != ["batch-0015"] or
            dev["threshold"] != choice["selected_threshold"]):
        raise ValueError("V11 development scores differ from amended selection")
    rows = load_approved(root, ["batch-0022"])
    if evaluate_pairs(rows, dev["predictions"]) != selected["report"]:
        raise ValueError("V11 development labels or order differ")
    return rows, dev["predictions"], choice


def main() -> None:
    if (ROOT / OUTPUT).exists():
        raise FileExistsError("Preserve prior v12 development choice")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit protocol, selector, score copy, and sealed test before selection")
    rows, items, choice = validated_inputs(ROOT)
    thresholds, report, counts = select_thresholds(rows, items)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            text=True).strip()
    record = {
        "status": "development_selection", "study": "v12_category_threshold_transfer",
        "source_commit": source_commit, "fit": "unchanged_v11_epoch_3",
        "model": choice["model"], "revision": choice["revision"],
        "selected_weight_sha256": choice["selected_weight_sha256"],
        "v11_selection_sha256": sha256(ROOT / V11_CHOICE),
        "development_batch": "batch-0022",
        "development_sha256": sha256(ROOT / "data/synthetic/batch-0022.jsonl"),
        "development_scores_sha256": sha256(ROOT / DEV_SCORES),
        "development_tag_amendment": "stored_batch_0015_tag_means_attested_batch_0022_rows_v11_pretest",
        "protocol_sha256": sha256(ROOT / "docs/V12_PROTOCOL.md"),
        "selector_sha256": sha256(Path(__file__)),
        "sealed_test_batch": "batch-0024",
        "sealed_test_sha256": sha256(ROOT / "data/synthetic/batch-0024.jsonl"),
        "thresholds": thresholds, "candidate_counts": counts,
        "selection_rule": "per_source_group_max_pairs_then_cards_then_fewer_false_reviews_then_nearest_half_then_lower_thresholds",
        "development_report": report,
    }
    (ROOT / OUTPUT).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps({"thresholds": thresholds, "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
