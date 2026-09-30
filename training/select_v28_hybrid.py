"""Freeze V28 hybrid cutoffs on development before prospective test reveal."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from eval.v26_metrics import CELLS, evaluate, select_thresholds
from roguard.mmbert_study import BASE_HASHES, MODEL_ID, REVISION
from roguard.review import sha256, taxonomy_sha256
from roguard.v28_hybrid import (NLI_HASHES, NLI_MODEL_ID, NLI_REVISION,
                                HybridComponents)
from training.generate_v28_hybrid import (ARTIFACT, COMMITMENT, DATA,
                                          PUBLIC_SEEDS, ROOT, build_rows)
from training.train_v25_synthetic import fit_char_baseline
from training.train_v26_facts import fact_report

PRIOR = ROOT / "models/bi-mmbert-v26-facts"


def verify_inputs(root: Path) -> tuple[dict, list[dict], list[dict], str]:
    root = root.resolve()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Commit V28 code, protocol, data, and seal before selecting cutoffs")
    record = json.loads((root / COMMITMENT).read_text(encoding="utf-8"))
    if (record.get("study") != "v28_bilingual_evidence_hybrid_synthetic" or
            record.get("status") != "sealed_same_origin_test_not_yet_generated" or
            record.get("test_rows_at_registration") != 0 or
            record.get("generator_sha256") != sha256(root / "training/generate_v28_hybrid.py") or
            record.get("source_dependency_sha256") != {
                name: sha256(root / name) for name in (
                    "training/generate_v25_synthetic.py",
                    "training/v25_ro.py", "training/v25_uk.py",
                    "training/v26_ro.py", "training/v26_uk.py",
                    "training/v28_ro.py", "training/v28_uk.py")
            } or record.get("taxonomy_sha256") != {
                lang: taxonomy_sha256(root, lang) for lang in ("ro", "uk")
            } or (root / DATA / "test.jsonl").exists()):
        raise ValueError("V28 pretest seal or source differs")
    rows = {}
    for split, seed in PUBLIC_SEEDS.items():
        path = root / DATA / f"{split}.jsonl"
        if sha256(path) != record[f"{split}_sha256"]:
            raise ValueError(f"V28 {split} hash differs")
        actual = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if actual != build_rows(split, seed):
            raise ValueError(f"V28 {split} does not reproduce the frozen generator")
        rows[split] = actual
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                            text=True).strip()
    return record, rows["train"], rows["dev"], source_commit


def select(mmbert_base: Path, nli_base: Path, root: Path = ROOT) -> dict:
    root = root.resolve()
    record, train, dev, source_commit = verify_inputs(root)
    artifact_path = root / ARTIFACT
    if artifact_path.exists():
        raise FileExistsError("Preserve any existing V28 hybrid selection")
    components = HybridComponents(mmbert_base, nli_base, root / "models/bi-mmbert-v26-facts")
    scores, evidence = components.score_cards(dev)
    thresholds = select_thresholds(dev, scores)
    report = evaluate(dev, scores, thresholds)
    facts_report = fact_report(dev, [item["composed_fact_scores"] for item in evidence])
    baseline, _ = fit_char_baseline(train, dev)
    baseline["report"] = evaluate(dev, baseline["scores"], baseline["thresholds"])
    eligible = (report["exact_pairs"] >= 168 and
                all(report["cells"][cell]["exact_pairs"] >= 40 for cell in CELLS))
    selection = {
        "study": "v28_bilingual_evidence_hybrid_synthetic",
        "status": "development_selection_before_test_reveal",
        "source_commit": source_commit,
        "test_seed_sha256": record["test_seed_sha256"],
        "test_seed_commitment_sha256": sha256(root / COMMITMENT),
        "train_sha256": record["train_sha256"],
        "dev_sha256": record["dev_sha256"],
        "taxonomy_sha256": record["taxonomy_sha256"],
        "hybrid_source_sha256": sha256(root / "src/roguard/v28_hybrid.py"),
        "selector_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(root / "eval/v26_metrics.py"),
        "protocol_sha256": sha256(root / "docs/V28_HYBRID_PROTOCOL.md"),
        "nli_model_id": NLI_MODEL_ID, "nli_revision": NLI_REVISION,
        "nli_files_sha256": NLI_HASHES,
        "mmbert_model_id": MODEL_ID, "mmbert_revision": REVISION,
        "mmbert_base_files_sha256": BASE_HASHES,
        "prior_artifact_manifest_sha256": sha256(PRIOR / "research.json"),
        "prior_weight_sha256": sha256(PRIOR / "adapter_model.safetensors"),
        "thresholds": thresholds,
        "scores": scores, "evidence": evidence,
        "report": report, "fact_report": facts_report,
        "char_baseline_dev": baseline,
        "development_eligible_for_test_reveal": eligible,
    }
    artifact_path.mkdir(parents=True)
    selection_path = artifact_path / "development-selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "study": "v28_bilingual_evidence_hybrid_synthetic",
        "status": "selected_before_test_reveal",
        "scope": "invented_abstract_metadata_only",
        "source_commit": source_commit,
        "test_seed_sha256": record["test_seed_sha256"],
        "test_seed_commitment_sha256": sha256(root / COMMITMENT),
        "train_sha256": record["train_sha256"], "dev_sha256": record["dev_sha256"],
        "thresholds": thresholds,
        "development_selection_sha256": sha256(selection_path),
        "development_eligible_for_test_reveal": eligible,
        "hybrid_source_sha256": sha256(root / "src/roguard/v28_hybrid.py"),
        "selector_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(root / "eval/v26_metrics.py"),
        "protocol_sha256": sha256(root / "docs/V28_HYBRID_PROTOCOL.md"),
        "nli_model_id": NLI_MODEL_ID, "nli_revision": NLI_REVISION,
        "nli_files_sha256": NLI_HASHES,
        "mmbert_model_id": MODEL_ID, "mmbert_revision": REVISION,
        "mmbert_base_files_sha256": BASE_HASHES,
        "prior_artifact_manifest_sha256": sha256(PRIOR / "research.json"),
        "prior_weight_sha256": sha256(PRIOR / "adapter_model.safetensors"),
        "independent_language_review": False,
        "eligible_for_live_child_message_screening": False,
    }
    (artifact_path / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return {"artifact": str(ARTIFACT), "development_pairs": report["exact_pairs"],
            "development_worst_cell": min(report["cells"][cell]["exact_pairs"]
                                          for cell in CELLS),
            "development_eligible_for_test_reveal": eligible,
            "test_rows_generated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmbert-base", type=Path, required=True)
    parser.add_argument("--nli-base", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select(args.mmbert_base, args.nli_base), indent=2))


if __name__ == "__main__":
    main()
