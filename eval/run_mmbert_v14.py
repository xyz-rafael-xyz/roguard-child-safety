"""Frozen mixed-input comparison of exact contracts and v11 advisory scores."""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
from pathlib import Path

from roguard import MMBertResearchClassifier, screen
from roguard.cli import assess_json
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256
from compare_qwen_balanced_v8 import pair_hits, paired_discordance

ROOT = Path(__file__).resolve().parents[1]
BATCH = "batch-0026"
ARTIFACT = Path("models/ro-mmbert-v11-abstract")
PROTOCOL = Path("docs/V14_PROTOCOL.md")
SIDECAR = Path("data/synthetic/batch-0026.contracts.jsonl")


def committed(root: Path, path: Path) -> None:
    saved = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], cwd=root,
                           check=True, capture_output=True).stdout
    if (root / path).read_bytes() != saved:
        raise ValueError(f"Commit frozen study input before inference: {path}")


def score_strata(rows: list[dict], items: list[dict]) -> dict[str, dict]:
    # Pair membership is decided from the positive row, so both sides stay together.
    advisory_ids = {row["id"] for left, right in zip(rows[::2], rows[1::2])
                    if right["labels"][0] in {"D1", "S1"} for row in (left, right)}
    by_id = {item["id"]: item for item in items}
    advisory_rows = [row for row in rows if row["id"] in advisory_ids]
    contract_rows = [row for row in rows if row["id"] not in advisory_ids]
    return {
        "advisory": evaluate_pairs(advisory_rows, [by_id[row["id"]] for row in advisory_rows]),
        "declared_contract": evaluate_pairs(contract_rows, [by_id[row["id"]] for row in contract_rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = ROOT.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve prior v14 test run")
    for path in (PROTOCOL, Path("training/generate_hybrid_holdout.py"),
                 Path(f"data/synthetic/{BATCH}.jsonl"), SIDECAR,
                 Path(f"data/synthetic/{BATCH}.review.json"),
                 Path(f"data/synthetic/{BATCH}.audit.json"),
                 ARTIFACT / "adapter_model.safetensors", ARTIFACT / "research.json"):
        committed(root, path)
    rows = load_approved(root, [BATCH])
    build_contracts = runpy.run_path(str(root / "training/generate_hybrid_holdout.py"))["build_contracts"]
    sidecars = [json.loads(line) for line in (root / SIDECAR).read_text(
        encoding="utf-8").splitlines()]
    audit = json.loads((root / f"data/synthetic/{BATCH}.audit.json").read_text(encoding="utf-8"))
    if (len(rows) != 144 or sidecars != build_contracts(BATCH) or
            audit.get("contract_sidecar_sha256") != sha256(root / SIDECAR) or
            any(item["id"] != row["id"] for item, row in zip(sidecars, rows))):
        raise ValueError("Frozen hybrid rows or sidecar differ")
    backend = MMBertResearchClassifier(args.base_model_path, root / ARTIFACT)
    model_items, hybrid_items = [], []
    for row, sidecar in zip(rows, sidecars):
        result = screen(row["text"], language="ro", source_kind=row["source_kind"],
                        backend=backend, thresholds=backend.thresholds)
        model_item = {"id": row["id"], "expected": row["labels"],
                      "predicted": list(result.labels), "strict_parse": True,
                      "scores": {code: result.scores[code] for code in
                                 applicable_codes(row["source_kind"])}}
        model_items.append(model_item)
        if sidecar["contract"] is None:
            hybrid_items.append({**model_item, "decision_basis": "abstract_model_advisory"})
        else:
            declared = assess_json({"language": "ro", **sidecar["contract"]})
            labels = [finding["category"] for finding in declared["findings"]
                      if finding["status"] != "pass"]
            hybrid_items.append({"id": row["id"], "expected": row["labels"],
                                 "predicted": labels, "strict_parse": True,
                                 "decision_basis": "caller_declared_contract"})
    hybrid_report = evaluate_pairs(rows, hybrid_items)
    model_report = evaluate_pairs(rows, model_items)
    strata = score_strata(rows, hybrid_items)
    by_code = {item["category"]: item for item in hybrid_report["per_category"]}
    success = (hybrid_report["exact_pairs"] >= 66 and
               strata["advisory"]["exact_pairs"] >= 18 and
               strata["declared_contract"]["exact_pairs"] == 48 and
               hybrid_report["false_review_on_negatives"] <= 4 and
               all(by_code[code]["tp"] >= 9 for code in by_code))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "frozen_v14_mixed_input_test", "batch": BATCH,
        "test_sha256": sha256(root / f"data/synthetic/{BATCH}.jsonl"),
        "sidecar_sha256": sha256(root / SIDECAR),
        "attestation_sha256": sha256(root / f"data/synthetic/{BATCH}.review.json"),
        "protocol_sha256": sha256(root / PROTOCOL),
        "adapter_weight_sha256": sha256(root / ARTIFACT / "adapter_model.safetensors"),
        "artifact_manifest_sha256": sha256(root / ARTIFACT / "research.json"),
        "threshold": backend.thresholds["D1"],
        "hybrid_predictions": hybrid_items, "model_predictions": model_items,
        "hybrid_report": hybrid_report, "text_only_report": model_report,
        "hybrid_strata": strata,
        "paired_discordance": paired_discordance(
            pair_hits(rows, {"predictions": hybrid_items}),
            pair_hits(rows, {"predictions": model_items})),
        "preregistered_success": success,
        "limitation": "Privileged declared contracts plus abstract cards; no authentic language validity.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"hybrid": hybrid_report, "text_only": model_report,
                      "strata": strata, "preregistered_success": success},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
