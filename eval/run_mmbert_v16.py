"""Compare frozen V16 and unchanged v11 on sealed abstract batch 0030."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import subprocess
from pathlib import Path

from roguard import MMBertResearchClassifier, screen
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
BATCH = "batch-0030"
PROTOCOL = Path("docs/V16_PROTOCOL.md")
V16 = Path("models/ro-mmbert-v16-abstract")
V11 = Path("models/ro-mmbert-v11-abstract")


def committed(root: Path, path: Path) -> None:
    saved = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], cwd=root,
                           check=True, capture_output=True).stdout
    if (root / path).read_bytes() != saved:
        raise ValueError(f"Commit frozen study input before inference: {path}")


def score_rows(rows: list[dict], backend) -> list[dict]:
    items = []
    for row in rows:
        result = screen(row["text"], language="ro", source_kind=row["source_kind"],
                        backend=backend, thresholds=backend.thresholds)
        codes = applicable_codes(row["source_kind"])
        items.append({"id": row["id"], "expected": row["labels"],
                      "predicted": list(result.labels), "strict_parse": True,
                      "scores": {code: result.scores[code] for code in codes}})
    return items


def discordance(rows: list[dict], candidate: list[dict], comparator: list[dict]) -> dict:
    counts = [0, 0]
    for index in range(0, len(rows), 2):
        expected = [row["labels"] for row in rows[index:index + 2]]
        left = all(item["predicted"] == label for item, label in
                   zip(candidate[index:index + 2], expected))
        right = all(item["predicted"] == label for item, label in
                    zip(comparator[index:index + 2], expected))
        counts[0] += left and not right
        counts[1] += right and not left
    discordant = sum(counts)
    probability = (min(1.0, 2 * sum(math.comb(discordant, k)
                                   for k in range(min(counts) + 1)) / 2 ** discordant)
                   if discordant else 1.0)
    return {"candidate_only_exact_pairs": counts[0],
            "comparator_only_exact_pairs": counts[1],
            "two_sided_exact_mcnemar_p": probability}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = ROOT.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior V16 test run")
    inputs = (PROTOCOL, Path("docs/V16_PREFIT_AMENDMENT.md"),
              Path("training/generate_advisory_repair.py"),
              Path("training/train_mmbert_advisory_repair.py"),
              Path(f"data/synthetic/{BATCH}.jsonl"),
              Path(f"data/synthetic/{BATCH}.review.json"),
              Path(f"data/synthetic/{BATCH}.audit.json"),
              Path("eval/runs/ro-mmbert-v16-dev-selection.json"),
              Path("eval/runs/ro-mmbert-v16-epoch-1-dev.json"),
              V16 / "adapter_model.safetensors", V16 / "research.json",
              V11 / "adapter_model.safetensors", V11 / "research.json")
    for path in inputs:
        committed(root, path)
    selected = runpy.run_path(str(root / "eval/verify_committed_v16_selection.py"))[
        "verify"](root)
    rows = load_approved(root, [BATCH])
    if (len(rows) != 96 or any(row["split"] != "test" for row in rows) or
            {row["labels"][0] for row in rows if row["labels"]} != {"D1", "S1"}):
        raise ValueError("Frozen V16 test batch differs")
    candidate = MMBertResearchClassifier(args.base_model_path, root / V16)
    comparator = MMBertResearchClassifier(args.base_model_path, root / V11)
    candidate_items = score_rows(rows, candidate)
    comparator_items = score_rows(rows, comparator)
    candidate_report = evaluate_pairs(rows, candidate_items)
    comparator_report = evaluate_pairs(rows, comparator_items)
    by_code = {entry["category"]: entry for entry in candidate_report["per_category"]}
    success = (candidate_report["exact_pairs"] >= 42 and
               candidate_report["false_review_on_negatives"] <= 4 and
               all(by_code[code]["tp"] >= 21 for code in ("D1", "S1")) and
               candidate_report["exact_pairs"] > comparator_report["exact_pairs"])
    always = [{"id": row["id"], "predicted": ["D1" if row["source_kind"] == "message" else "S1"],
               "strict_parse": True} for row in rows]
    never = [{"id": row["id"], "predicted": [], "strict_parse": True} for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "frozen_v16_advisory_test", "batch": BATCH,
        "test_sha256": sha256(root / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(root / f"data/synthetic/{BATCH}.review.json"),
        "protocol_sha256": sha256(root / PROTOCOL),
        "prefit_amendment_sha256": sha256(root / "docs/V16_PREFIT_AMENDMENT.md"),
        "selection_record_sha256": sha256(root / "eval/runs/ro-mmbert-v16-dev-selection.json"),
        "v16_manifest_sha256": sha256(root / V16 / "research.json"),
        "v11_manifest_sha256": sha256(root / V11 / "research.json"),
        "v16_weight_sha256": sha256(root / V16 / "adapter_model.safetensors"),
        "v11_weight_sha256": sha256(root / V11 / "adapter_model.safetensors"),
        "v16_threshold": candidate.thresholds["D1"],
        "v11_threshold": comparator.thresholds["D1"],
        "selected_development": selected,
        "candidate_predictions": candidate_items,
        "comparator_predictions": comparator_items,
        "candidate_report": candidate_report,
        "comparator_report": comparator_report,
        "paired_discordance": discordance(rows, candidate_items, comparator_items),
        "always_review_reference": evaluate_pairs(rows, always),
        "never_review_reference": evaluate_pairs(rows, never),
        "preregistered_success": success,
        "limitation": "Abstract D1/S1 prose cards only; no authentic child-language validity.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_report": candidate_report,
                      "comparator_report": comparator_report,
                      "preregistered_success": success}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
