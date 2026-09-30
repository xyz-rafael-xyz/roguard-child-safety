"""Score the once-revealed V27 balanced fact model against frozen comparators."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from eval.v26_metrics import CELLS, evaluate, target_passed
from roguard.review import sha256, taxonomy_sha256
from roguard.v25_abstract import V25SyntheticResearchClassifier
from roguard.v26_facts import V26FactResearchClassifier
from roguard.v27_balanced import V27BalancedResearchClassifier
from training.generate_v27_balanced import (ARTIFACT, COMMITMENT, DATA, PRIVATE_SEED,
                                         PUBLIC_SEEDS, ROOT, build_rows)
from training.train_v25_synthetic import char_baseline_model, char_baseline_scores
from training.train_v27_balanced import fact_report

RESULT = Path("eval/prospective/v27-balanced-result.json")
PREDICTIONS = Path("eval/prospective/v27-balanced-predictions.jsonl")
PRIOR_ARTIFACT = Path("models/bi-mmbert-v25-abstract")
SECOND_PRIOR_ARTIFACT = Path("models/bi-mmbert-v26-facts")


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
    if (commitment.get("study") != "v27_bilingual_balanced_fact_synthetic" or
            commitment.get("status") != "sealed_same_origin_test_not_yet_generated" or
            commitment.get("test_rows_at_registration") != 0 or
            manifest.get("status") != "selected_before_test_reveal" or
            manifest.get("study") != commitment["study"] or
            manifest.get("development_eligible_for_test_reveal") is not True or
            selection.get("development_eligible_for_test_reveal") is not True or
            manifest.get("test_seed_sha256") != commitment.get("test_seed_sha256") or
            manifest.get("test_seed_commitment_sha256") != sha256(root / COMMITMENT) or
            manifest.get("development_selection_sha256") != sha256(selection_path) or
            manifest.get("selected_epoch") != selection.get("selected_epoch") or
            manifest.get("thresholds") != selection.get("selected_thresholds") or
            manifest.get("weight_sha256") != selection.get("selected_weight_sha256") or
            manifest.get("prompt_sha256") != sha256(root / "src/roguard/v26_facts.py") or
            manifest.get("loader_sha256") != sha256(root / "src/roguard/v27_balanced.py") or
            manifest.get("weighting_sha256") != sha256(root / "training/v27_weights.py") or
            manifest.get("trainer_sha256") != sha256(root / "training/train_v27_balanced.py") or
            manifest.get("metrics_sha256") != sha256(root / "eval/v26_metrics.py") or
            manifest.get("protocol_sha256") != sha256(root / "docs/V27_BALANCED_PROTOCOL.md") or
            commitment.get("generator_sha256") != sha256(root / "training/generate_v27_balanced.py") or
            commitment.get("source_dependency_sha256") != {
                name: sha256(root / name) for name in (
                    "training/generate_v25_synthetic.py",
                    "training/generate_v26_facts.py",
                    "training/v25_ro.py", "training/v25_uk.py",
                    "training/v26_ro.py", "training/v26_uk.py")
            } or commitment.get("taxonomy_sha256") != {
                lang: taxonomy_sha256(root, lang) for lang in ("ro", "uk")
            } or set(manifest.get("thresholds", {})) != set(CELLS)):
        raise ValueError("V27 pretest seal, selected model, or source differs")
    for relative in (COMMITMENT, ARTIFACT / "research.json",
                     ARTIFACT / "development-selection.json",
                     ARTIFACT / "adapter_model.safetensors"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", "--", str(relative)],
                          cwd=root, capture_output=True, check=False).returncode or subprocess.run(
                              ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
                              cwd=root, capture_output=True, check=False).returncode:
            raise ValueError(f"V27 selected artifact is not frozen in Git: {relative}")
    seed = (root / PRIVATE_SEED).read_text(encoding="ascii").strip()
    if hashlib.sha256(seed.encode()).hexdigest() != commitment["test_seed_sha256"]:
        raise ValueError("V27 test seed differs from the prefit seal")
    rows = {}
    for split in ("train", "dev", "test"):
        path = root / DATA / f"{split}.jsonl"
        actual = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        split_seed = seed if split == "test" else PUBLIC_SEEDS[split]
        if actual != build_rows(split, split_seed):
            raise ValueError(f"V27 {split} is not the frozen generator output")
        if split != "test" and sha256(path) != commitment[f"{split}_sha256"]:
            raise ValueError(f"V27 {split} hash differs from the seal")
        rows[split] = actual
    if [len(rows[key]) for key in ("train", "dev", "test")] != [1536, 384, 384]:
        raise ValueError("V27 split sizes differ")
    return commitment, manifest, selection, rows["train"], rows["dev"], rows["test"]


def run(base: Path, root: Path = ROOT) -> dict:
    root = root.resolve()
    if (root / RESULT).exists() or (root / PREDICTIONS).exists():
        raise FileExistsError("Preserve the existing V27 sealed test report")
    commitment, manifest, selection, train, dev, test = verify_frozen_study(root)
    classifier = V27BalancedResearchClassifier(base, root / ARTIFACT)
    dev_scores, dev_facts = classifier.score_cards(dev)
    selected = next(item for item in selection["candidates"]
                    if item["epoch"] == selection["selected_epoch"])
    if (dev_scores != selected["scores"] or dev_facts != selected["fact_scores"] or
            evaluate(dev, dev_scores, manifest["thresholds"]) != selected["report"]):
        raise ValueError("V27 saved model does not reproduce its selected development results")
    scores, fact_scores = classifier.score_cards(test)
    model_report = evaluate(test, scores, manifest["thresholds"])
    baseline_model = char_baseline_model(train)
    baseline_dev_scores = char_baseline_scores(baseline_model, dev)
    baseline = selection["char_baseline_dev"]
    if baseline_dev_scores != baseline["scores"]:
        raise ValueError("V27 lexical comparator differs from the pretest selection")
    baseline_scores = char_baseline_scores(baseline_model, test)
    baseline_report = evaluate(test, baseline_scores, baseline["thresholds"])
    prior = V25SyntheticResearchClassifier(base, root / PRIOR_ARTIFACT)
    prior_scores = prior.score_cards(test)
    prior_report = evaluate(test, prior_scores, prior.thresholds)
    second_prior = V26FactResearchClassifier(base, root / SECOND_PRIOR_ARTIFACT)
    second_prior_scores, _ = second_prior.score_cards(test)
    second_prior_report = evaluate(test, second_prior_scores, second_prior.thresholds)
    predictions = b"".join(_jsonl({
        "id": row["id"], "expected": row["labels"],
        "model_score": score, "fact_scores": facts,
        "model_predicted": ["D1" if row["source_kind"] == "message" else "S1"]
                           if score >= manifest["thresholds"][
                               f"{row['language']}:{'D1' if row['source_kind'] == 'message' else 'S1'}"]
                           else [],
        "char_baseline_score": baseline_score, "v25_model_score": prior_score,
        "v26_model_score": second_prior_score,
    }) for row, score, facts, baseline_score, prior_score, second_prior_score in zip(
        test, scores, fact_scores, baseline_scores, prior_scores, second_prior_scores))
    result = {
        "study": "v27_bilingual_balanced_fact_synthetic",
        "scope": "invented_abstract_metadata_only",
        "independent_human_authorship": False,
        "independent_language_review": False,
        "eligible_for_live_child_message_screening": False,
        "test_seed_sha256": commitment["test_seed_sha256"],
        "test_sha256": sha256(root / DATA / "test.jsonl"),
        "selected_model_manifest_sha256": sha256(root / ARTIFACT / "research.json"),
        "selected_weight_sha256": manifest["weight_sha256"],
        "predictions_sha256": hashlib.sha256(predictions).hexdigest(),
        "model": model_report, "latent_fact_accuracy": fact_report(test, fact_scores),
        "fixed_target_passed": target_passed(model_report),
        "char_ngram_baseline": baseline_report,
        "frozen_v25_model": prior_report,
        "frozen_v26_model": second_prior_report,
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
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.base_model_path), indent=2))


if __name__ == "__main__":
    main()
