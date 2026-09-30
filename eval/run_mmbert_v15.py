"""One frozen D1/S1 inference on sealed Romanian abstract prose batch 0027."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from roguard import MMBertResearchClassifier, screen
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
BATCH = "batch-0027"
ARTIFACT = Path("models/ro-mmbert-v11-abstract")
PROTOCOL = Path("docs/V15_PROTOCOL.md")


def committed(root: Path, path: Path) -> None:
    saved = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], cwd=root,
                           check=True, capture_output=True).stdout
    if (root / path).read_bytes() != saved:
        raise ValueError(f"Commit frozen study input before inference: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = ROOT.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior v15 test run")
    for path in (PROTOCOL, Path("training/generate_advisory_prose_holdout.py"),
                 Path(f"data/synthetic/{BATCH}.jsonl"), Path(f"data/synthetic/{BATCH}.review.json"),
                 Path(f"data/synthetic/{BATCH}.audit.json"),
                 ARTIFACT / "adapter_model.safetensors", ARTIFACT / "research.json"):
        committed(root, path)
    rows = load_approved(root, [BATCH])
    if (len(rows) != 96 or any(row["split"] != "test" for row in rows) or
            {row["labels"][0] for row in rows if row["labels"]} != {"D1", "S1"}):
        raise ValueError("Frozen advisory prose batch differs")
    backend = MMBertResearchClassifier(args.base_model_path, root / ARTIFACT)
    items = []
    for row in rows:
        result = screen(row["text"], language="ro", source_kind=row["source_kind"],
                        backend=backend, thresholds=backend.thresholds)
        items.append({"id": row["id"], "expected": row["labels"],
                      "predicted": list(result.labels), "strict_parse": True,
                      "scores": {code: result.scores[code] for code in result.scores
                                 if row["source_kind"] == "response" and code in ("A1", "S1")
                                 or row["source_kind"] == "message" and code == "D1"}})
    report = evaluate_pairs(rows, items)
    by_code = {entry["category"]: entry for entry in report["per_category"]}
    success = (report["exact_pairs"] >= 42 and
               report["false_review_on_negatives"] <= 4 and
               all(by_code[code]["tp"] >= 21 for code in ("D1", "S1")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "frozen_v15_advisory_prose_test", "batch": BATCH,
        "test_sha256": sha256(root / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(root / f"data/synthetic/{BATCH}.review.json"),
        "protocol_sha256": sha256(root / PROTOCOL),
        "adapter_weight_sha256": sha256(root / ARTIFACT / "adapter_model.safetensors"),
        "artifact_manifest_sha256": sha256(root / ARTIFACT / "research.json"),
        "threshold": backend.thresholds["D1"],
        "predictions": items, "report": report,
        "always_review_reference": {"exact_cards": 48, "exact_pairs": 0,
                                    "false_review_on_negatives": 48},
        "never_review_reference": {"exact_cards": 48, "exact_pairs": 0,
                                   "false_review_on_negatives": 0},
        "preregistered_success": success,
        "limitation": "Abstract D1/S1 descriptors only; no authentic child-language validity.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": report, "preregistered_success": success},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
