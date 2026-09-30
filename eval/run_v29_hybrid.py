"""Score the once-revealed V29 anchor-routed hybrid without reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from eval.v26_metrics import CELLS, evaluate, target_passed
from roguard.review import sha256, taxonomy_sha256
from roguard.v25_abstract import V25SyntheticResearchClassifier
from roguard.v29_hybrid import V29HybridResearchClassifier
from training.generate_v29_hybrid import (ARTIFACT, COMMITMENT, DATA, PRIVATE_SEED,
                                          PUBLIC_SEEDS, ROOT, build_rows)
from training.train_v25_synthetic import char_baseline_model, char_baseline_scores
from training.train_v26_facts import fact_report

RESULT = Path("eval/prospective/v29-hybrid-result.json")
PREDICTIONS = Path("eval/prospective/v29-hybrid-predictions.jsonl")
PRIOR = Path("models/bi-mmbert-v26-facts")
FIRST_PRIOR = Path("models/bi-mmbert-v25-abstract")


def _json(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _jsonl(data: dict) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def verify_frozen_study(root: Path) -> tuple[dict, dict, dict, list[dict], list[dict], list[dict]]:
    root = root.resolve()
    commitment = json.loads((root / COMMITMENT).read_text(encoding="utf-8"))
    manifest_path = root / ARTIFACT / "research.json"
    selection_path = root / ARTIFACT / "development-selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (commitment.get("study") != "v29_bilingual_anchor_routed_synthetic" or
            commitment.get("status") != "sealed_same_origin_test_not_yet_generated" or
            commitment.get("test_rows_at_registration") != 0 or
            manifest.get("status") != "selected_before_test_reveal" or
            manifest.get("study") != commitment["study"] or
            manifest.get("development_eligible_for_test_reveal") is not True or
            selection.get("development_eligible_for_test_reveal") is not True or
            manifest.get("test_seed_sha256") != commitment.get("test_seed_sha256") or
            manifest.get("test_seed_commitment_sha256") != sha256(root / COMMITMENT) or
            manifest.get("development_selection_sha256") != sha256(selection_path) or
            manifest.get("thresholds") != selection.get("thresholds") or
            manifest.get("hybrid_source_sha256") != sha256(root / "src/roguard/v29_hybrid.py") or
            manifest.get("parent_hybrid_source_sha256") != sha256(root / "src/roguard/v28_hybrid.py") or
            manifest.get("selector_sha256") != sha256(root / "training/select_v29_hybrid.py") or
            manifest.get("metrics_sha256") != sha256(root / "eval/v26_metrics.py") or
            manifest.get("protocol_sha256") != sha256(root / "docs/V29_HYBRID_PROTOCOL.md") or
            manifest.get("prior_artifact_manifest_sha256") != sha256(root / PRIOR / "research.json") or
            manifest.get("prior_weight_sha256") != sha256(root / PRIOR / "adapter_model.safetensors") or
            commitment.get("generator_sha256") != sha256(root / "training/generate_v29_hybrid.py") or
            commitment.get("source_dependency_sha256") != {
                name: sha256(root / name) for name in (
                    "training/generate_v25_synthetic.py",
                    "training/v25_ro.py", "training/v25_uk.py",
                    "training/v26_ro.py", "training/v26_uk.py",
                    "training/v28_ro.py", "training/v28_uk.py",
                    "training/v29_ro.py", "training/v29_uk.py")
            } or commitment.get("taxonomy_sha256") != {
                lang: taxonomy_sha256(root, lang) for lang in ("ro", "uk")
            } or set(manifest.get("thresholds", {})) != set(CELLS)):
        raise ValueError("V29 pretest seal, selected hybrid, or source differs")
    for relative in (COMMITMENT, ARTIFACT / "research.json",
                     ARTIFACT / "development-selection.json"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(relative)],
                          cwd=root, capture_output=True, check=False).returncode or subprocess.run(
                              ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
                              cwd=root, capture_output=True, check=False).returncode:
            raise ValueError(f"V29 hybrid selection is not frozen in Git: {relative}")
    seed = (root / PRIVATE_SEED).read_text(encoding="ascii").strip()
    if hashlib.sha256(seed.encode()).hexdigest() != commitment["test_seed_sha256"]:
        raise ValueError("V29 test seed differs from the prefit seal")
    rows = {}
    for split in ("train", "dev", "test"):
        path = root / DATA / f"{split}.jsonl"
        actual = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        split_seed = seed if split == "test" else PUBLIC_SEEDS[split]
        if actual != build_rows(split, split_seed):
            raise ValueError(f"V29 {split} is not the frozen generator output")
        if split != "test" and sha256(path) != commitment[f"{split}_sha256"]:
            raise ValueError(f"V29 {split} hash differs from the seal")
        rows[split] = actual
    if [len(rows[key]) for key in ("train", "dev", "test")] != [1536, 384, 384]:
        raise ValueError("V29 split sizes differ")
    return commitment, manifest, selection, rows["train"], rows["dev"], rows["test"]


def run(mmbert_base: Path, nli_base: Path, root: Path = ROOT) -> dict:
    root = root.resolve()
    if (root / RESULT).exists() or (root / PREDICTIONS).exists():
        raise FileExistsError("Preserve the existing V29 sealed test report")
    commitment, manifest, selection, train, dev, test = verify_frozen_study(root)
    classifier = V29HybridResearchClassifier(mmbert_base, nli_base,
                                              root / PRIOR, root / ARTIFACT)
    dev_scores, dev_evidence = classifier.score_cards(dev)
    if (dev_scores != selection["scores"] or dev_evidence != selection["evidence"] or
            evaluate(dev, dev_scores, manifest["thresholds"]) != selection["report"]):
        raise ValueError("V29 hybrid does not reproduce its selected development result")
    scores, evidence = classifier.score_cards(test)
    model_report = evaluate(test, scores, manifest["thresholds"])
    baseline_model = char_baseline_model(train)
    baseline_dev = char_baseline_scores(baseline_model, dev)
    baseline = selection["char_baseline_dev"]
    if baseline_dev != baseline["scores"]:
        raise ValueError("V29 lexical comparator differs from pretest selection")
    baseline_scores = char_baseline_scores(baseline_model, test)
    baseline_report = evaluate(test, baseline_scores, baseline["thresholds"])
    prior_scores, _ = classifier.fact_model.score_cards(test)
    prior_report = evaluate(test, prior_scores, classifier.fact_model.thresholds)
    first_prior = V25SyntheticResearchClassifier(mmbert_base, root / FIRST_PRIOR)
    first_prior_scores = first_prior.score_cards(test)
    first_prior_report = evaluate(test, first_prior_scores, first_prior.thresholds)
    predictions = b"".join(_jsonl({
        "id": row["id"], "expected": row["labels"],
        "model_score": score, "evidence": detail,
        "model_predicted": ["D1" if row["source_kind"] == "message" else "S1"]
                           if score >= manifest["thresholds"][
                               f"{row['language']}:{'D1' if row['source_kind'] == 'message' else 'S1'}"]
                           else [],
        "char_baseline_score": baseline_score,
        "v25_model_score": first_score, "v26_model_score": prior_score,
    }) for row, score, detail, baseline_score, first_score, prior_score in zip(
        test, scores, evidence, baseline_scores, first_prior_scores, prior_scores))
    result = {
        "study": "v29_bilingual_anchor_routed_synthetic",
        "scope": "invented_abstract_metadata_only",
        "independent_human_authorship": False,
        "independent_language_review": False,
        "eligible_for_live_child_message_screening": False,
        "test_seed_sha256": commitment["test_seed_sha256"],
        "test_sha256": sha256(root / DATA / "test.jsonl"),
        "selected_hybrid_manifest_sha256": sha256(root / ARTIFACT / "research.json"),
        "predictions_sha256": hashlib.sha256(predictions).hexdigest(),
        "model": model_report,
        "latent_fact_accuracy": fact_report(
            test, [item["composed_fact_scores"] for item in evidence]),
        "fixed_target_passed": target_passed(model_report),
        "char_ngram_baseline": baseline_report,
        "frozen_v25_model": first_prior_report,
        "frozen_v26_model": prior_report,
        "always_review": evaluate(test, [1.0] * len(test), {cell: 0.5 for cell in CELLS}),
        "never_review": evaluate(test, [0.0] * len(test), {cell: 0.5 for cell in CELLS}),
    }
    (root / PREDICTIONS).parent.mkdir(parents=True, exist_ok=True)
    with (root / PREDICTIONS).open("xb") as stream:
        stream.write(predictions)
    with (root / RESULT).open("xb") as stream:
        stream.write(_json(result))
    return {"result": str(RESULT), "test_pairs": model_report["pairs"],
            "exact_pairs": model_report["exact_pairs"],
            "fixed_target_passed": result["fixed_target_passed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmbert-base", type=Path, required=True)
    parser.add_argument("--nli-base", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.mmbert_base, args.nli_base), indent=2))


if __name__ == "__main__":
    main()
