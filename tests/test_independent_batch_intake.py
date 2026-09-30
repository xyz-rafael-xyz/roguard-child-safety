"""Structural fixtures for independent abstract intake; no human review is implied."""

import json
import subprocess
import os
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from roguard.blind_packets import build_blind_packets
from roguard.human_eval import evaluate_adjudicated
from roguard.human_eval_registered import evaluate_adjudicated_registered
from roguard.human_eval import _read_blind_predictions
from roguard.independent_check import check_independent_batch
from roguard.packet_binding import binding_path
from roguard.pair_eval import evaluate_pairs, main as pairs_main
from roguard.prospective_predict import freeze_packet_predictions
from roguard.review import (INDEPENDENT_FACTORS, ReviewError, load_approved,
                            sha256, taxonomy_sha256)

ROOT = Path(__file__).resolve().parents[1]
BATCH = "batch-0099"


def _fixture_decision(factor: str, positive: bool) -> dict:
    """Mechanical one-fact declarations for tests, never human judgments."""
    facts = {
        "source_role": "minor", "safety_or_support_anchor": True,
        "indirect_or_repeated_support_pattern": True,
        "explicit_support_request": False,
    }
    if factor == "explicit_support_request":
        facts["indirect_or_repeated_support_pattern"] = False
        facts["explicit_support_request"] = positive
    elif not positive:
        facts[factor] = "adult" if factor == "source_role" else False
    return {"language": "ro", "disclosure": facts}


def _fixture_support_decision(factor: str, positive: bool) -> dict:
    fields = set(INDEPENDENT_FACTORS["S1"])
    return {"language": "ro", "support": {
        "applicable_fields": sorted(fields),
        "passed_fields": sorted(fields - {factor} if positive else fields),
    }}


def fixture(root: Path, language: str = "ro", category: str = "D1") -> None:
    (root / "taxonomy").mkdir(parents=True)
    (root / "data/synthetic").mkdir(parents=True)
    shutil.copyfile(ROOT / f"taxonomy/taxonomy_{language}.md",
                    root / f"taxonomy/taxonomy_{language}.md")
    taxonomy_hash = taxonomy_sha256(root, language)
    review_name = "review_ro_independent.json" if language == "ro" else "review_uk.json"
    (root / "taxonomy" / review_name).write_text(json.dumps({
        "status": "approved", "taxonomy_sha256": taxonomy_hash,
        "reviews": [
            {"kind": kind, "reviewer": reviewer, "reviewed_at": "2026-09-30",
             "independent_of_author": True, "summary": "Fixture declaration only",
             "category_decisions": {code: "accept" for code in ("D1", "R1", "A1", "P1", "G1", "S1")}}
            for kind, reviewer in (("language", "fixture_lang"),
                                   ("child_safety", "fixture_safety"))
        ],
    }))
    adapter = root / "models/fixture-abstract"
    adapter.mkdir(parents=True)
    weight = adapter / "adapter_model.safetensors"
    weight.write_bytes(b"fixture weight, not a model")
    manifest = adapter / "research.json"
    manifest.write_text(json.dumps({"trained_language": language,
                                    "shared_threshold": 0.5,
                                    "adapter_weight_sha256": sha256(weight)}))
    source = root / "src/roguard"
    source.mkdir(parents=True)
    predictor = source / "fixture_predict.py"
    evaluator = source / "fixture_eval.py"
    predictor.write_text("# mechanical fixture\n")
    evaluator.write_text("# mechanical fixture\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                    "commit", "-qm", "fixture model"], cwd=root, check=True)
    model_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           text=True).strip()
    freeze_path = root / "eval/prospective/fixture-freeze.json"
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "registered_baseline_before_independent_batch",
        "registered_on": "2026-09-30", "pre_author_git_commit": model_commit,
        "language": language, "categories": ["D1", "S1"],
        "input_scope": "independently_authored_abstract_cards_only",
        "adapter_dir": "models/fixture-abstract", "model_id": "fixture", "cutoff": 0.5,
        "prediction_model_id": "fixture_only",
        "predictor_entrypoint": "src/roguard/fixture_predict.py",
        "evaluator_entrypoint": "src/roguard/fixture_eval.py",
        "sha256": {str(file.relative_to(root)): sha256(file)
                   for file in (weight, manifest, predictor, evaluator)},
    }))
    subprocess.run(["git", "add", "eval/prospective/fixture-freeze.json"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                    "commit", "-qm", "fixture freeze"], cwd=root, check=True)
    rows = []
    kind = "message" if category == "D1" else "response"
    for index in range(1, 49):
        for variant in (0, 1):
            rows.append({
                "id": f"{BATCH}-{language}-{len(rows) + 1:05d}", "language": language,
                "source_kind": kind, "split": "test",
                "text": ((f"Fișă abstractă {index:03d}. Indicatorul simbolic este "
                          f"{'prezent' if variant else 'absent'}. Fără comunicare redată.")
                         if language == "ro" else
                         (f"Умовна картка {index:03d}. Символічний показник "
                          f"{'наявний' if variant else 'відсутній'}. Мовлення не відтворено.")),
                "labels": [category] if variant else [],
                "origin": "independent_abstract_v1",
            })
    stem = root / "data/synthetic" / BATCH
    data = stem.with_suffix(".jsonl")
    preview = stem.with_suffix(".preview.md")
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    preview.write_text("# Structural fixture\n\n| ID | Limbă | Tip | Etichete | Text |\n"
                       "|---|---|---|---|---|\n" + "".join(
                           f"| {row['id']} | {language} | {kind} | {','.join(row['labels']) or 'none'} | {row['text']} |\n"
                           for row in rows))
    provenance = {
        "schema_version": 2, "batch": BATCH, "language": language, "category": category,
        "model_freeze_path": "eval/prospective/fixture-freeze.json",
        "model_freeze_sha256": sha256(freeze_path),
        "authors": [{"author_id": author, "native_language": language,
                     "independent_of_project": True, "authored_at": "2026-09-30"}
                    for author in ("fixture_author_a", "fixture_author_b")],
        "pair_authors": ["fixture_author_a"] * 24 + ["fixture_author_b"] * 24,
        "pair_factors": ([factor for factor in INDEPENDENT_FACTORS[category]
                          for _ in range(24 // len(INDEPENDENT_FACTORS[category]))] * 2),
        "safety_review": {"reviewer_id": "fixture_content_review", "reviewed_at": "2026-09-30",
                          "abstract_only_checked": True, "no_realistic_content_checked": True,
                          "pair_logic_checked": True, "factor_balance_checked": True,
                          "prior_overlap_checked": True},
        "label_status": "author_intended_not_adjudicated",
    }
    provenance_path = stem.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance))
    record = {
        "status": "approved", "reviewer": "fixture_content_review", "reviewed_at": "2026-09-30",
        "reviewed_ids": [row["id"] for row in rows], "language": language,
        "batch_sha256": sha256(data), "preview_sha256": sha256(preview),
        "taxonomy_sha256": taxonomy_hash, "provenance_sha256": sha256(provenance_path),
    }
    stem.with_suffix(".review.json").write_text(json.dumps(record))


def fixture_blind_predictions(packets: Path, root: Path, category: str) -> dict:
    """Build a mechanical prediction fixture; never treat it as a model result."""
    owner = json.loads((packets / "owner-map.json").read_text())
    rows = load_approved(root, [BATCH])
    labels = {row["id"]: row["labels"] for row in rows}
    source = {item["item_id"]: item["source_id"] for item in owner["id_map"]}
    packet = [json.loads(line) for line in (packets / "reviewer-a.jsonl").read_text().splitlines()]
    return {"schema_version": 2,
            "packet_sha256": owner["packet_sha256"]["reviewer-a.jsonl"],
            "language": owner["language"], "category": category, "model_id": "fixture_only",
            "predictions": [{"item_id": item["item_id"],
                             "predicted": labels[source[item["item_id"]]],
                             "strict_parse": True} for item in packet]}


def fixture_packet_binding(answers: Path, packet: Path) -> None:
    sidecar = binding_path(answers)
    sidecar.write_text(json.dumps({"schema_version": 1,
                                   "packet_sha256": sha256(packet)}))
    os.chmod(sidecar, 0o600)


def predicted_row_for_source(payload: dict, owner: dict, source_id: str) -> dict:
    opaque = next(item["item_id"] for item in owner["id_map"]
                  if item["source_id"] == source_id)
    return next(item for item in payload["predictions"] if item["item_id"] == opaque)


class IndependentBatchIntakeTests(unittest.TestCase):
    def test_frozen_predictions_use_only_opaque_packet_and_fixed_cutoff(self):
        class FixtureBackend:
            thresholds = {"D1": 0.5}

            def score(self, text, language, source_kind):
                if language != "ro" or source_kind != "message":
                    raise AssertionError("Unexpected input to fixture scorer")
                return {"D1": 0.9 if "prezent" in text else 0.1}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            packets = root / "private_packets"
            build_blind_packets(root, BATCH, "D1", packets)
            packet = packets / "reviewer-a.jsonl"
            output = root / "predictions.json"
            result = freeze_packet_predictions(packet, FixtureBackend(),
                                               "fixture_scoring_only", output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(payload), {"schema_version", "packet_sha256", "language",
                                            "category", "model_id", "predictions"})
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["packet_sha256"], sha256(packet))
            self.assertEqual(len(payload["predictions"]), 96)
            self.assertNotIn('"text"', output.read_text(encoding="utf-8"))
            self.assertNotIn('"labels"', output.read_text(encoding="utf-8"))
            owner = json.loads((packets / "owner-map.json").read_text())
            _, predictions = _read_blind_predictions(
                output, packet, sha256(packet), "ro", "D1",
                {item["item_id"]: item["source_id"] for item in owner["id_map"]})
            self.assertEqual(sum(predictions.values()), 48)
            self.assertFalse(result["source_batch_or_author_labels_read"])
            self.assertFalse(result["independent_human_review_verified_by_software"])
            original = output.read_text(encoding="utf-8")
            for damaged in (
                {**payload, "packet_sha256": "0" * 64},
                {**payload, "predictions": list(reversed(payload["predictions"]))},
                {**payload, "predictions": [
                    {**payload["predictions"][0], "expected": []},
                    *payload["predictions"][1:]]},
            ):
                output.write_text(json.dumps(damaged))
                with self.assertRaises(ValueError):
                    _read_blind_predictions(
                        output, packet, sha256(packet), "ro", "D1",
                        {item["item_id"]: item["source_id"] for item in owner["id_map"]})
            output.write_text(original)
            with self.assertRaises(FileExistsError):
                freeze_packet_predictions(packet, FixtureBackend(),
                                          "fixture_scoring_only", output)

    def test_packet_predictor_rejects_ukrainian_or_malformed_packets(self):
        class UnusedBackend:
            thresholds = {"D1": 0.5}

            def score(self, *args):
                raise AssertionError("Scorer must not run on a rejected source")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "predictions.json"
            malformed = root / "malformed.jsonl"
            malformed.write_text('{"item_id":"x"}\n')
            with self.assertRaises(ValueError):
                freeze_packet_predictions(malformed, UnusedBackend(),
                                          "fixture_scoring_only", output)
            fixture(root, language="uk")
            packets = root / "private_packets"
            build_blind_packets(root, BATCH, "D1", packets)
            with self.assertRaisesRegex(ValueError, "Romanian packet"):
                freeze_packet_predictions(packets / "reviewer-a.jsonl", UnusedBackend(),
                                          "fixture_scoring_only", output)
            self.assertFalse(output.exists())

    def test_structural_declarations_can_reach_blinded_packets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            rows = load_approved(root, [BATCH])
            self.assertEqual(len(rows), 96)
            checked = check_independent_batch(root, BATCH)
            self.assertEqual(checked["pairs"], 48)
            self.assertTrue(checked["declared_factor_coverage_structurally_checked"])
            self.assertFalse(checked["one_fact_contrast_verified_by_software"])
            self.assertFalse(checked["actual_author_independence_verified"])
            self.assertFalse(checked["intended_labels_adjudicated"])
            packets = root / "private_packets"
            result = build_blind_packets(root, BATCH, "D1", packets)
            self.assertEqual(result["cards"], 96)
            self.assertFalse(result["reviewer_independence_verified"])
            self.assertFalse(json.loads((packets / "owner-map.json").read_text())["label_validity_verified"])

    def test_ukrainian_s1_structural_fixture_reaches_blinded_packets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, language="uk", category="S1")
            checked = check_independent_batch(root, BATCH)
            self.assertEqual((checked["language"], checked["category"]), ("uk", "S1"))
            packets = root / "private_packets"
            result = build_blind_packets(root, BATCH, "S1", packets)
            self.assertEqual(result["cards"], 96)
            self.assertFalse(result["reviewer_independence_verified"])

    def test_manual_intake_cannot_relabel_romanian_card_as_ukrainian(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, language="uk", category="S1")
            base = root / "data/synthetic" / BATCH
            data, preview, review = (base.with_suffix(suffix) for suffix in
                                     (".jsonl", ".preview.md", ".review.json"))
            rows = [json.loads(line) for line in data.read_text().splitlines()]
            original = rows[0]["text"]
            rows[0]["text"] = "Tabel abstract cu rubrici simbolice, toate câmpurile rămân constante."
            data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            preview.write_text(preview.read_text().replace(original, rows[0]["text"], 1))
            attestation = json.loads(review.read_text())
            attestation.update(batch_sha256=sha256(data), preview_sha256=sha256(preview))
            review.write_text(json.dumps(attestation))
            with self.assertRaisesRegex(ReviewError, "language script"):
                check_independent_batch(root, BATCH)

    def test_manual_intake_applies_same_card_length_as_author_kit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            base = root / "data/synthetic" / BATCH
            data, preview, review = (base.with_suffix(suffix) for suffix in
                                     (".jsonl", ".preview.md", ".review.json"))
            rows = [json.loads(line) for line in data.read_text().splitlines()]
            original = rows[0]["text"]
            rows[0]["text"] = "Fișă scurtă"
            data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            preview.write_text(preview.read_text().replace(original, rows[0]["text"], 1))
            attestation = json.loads(review.read_text())
            attestation.update(batch_sha256=sha256(data), preview_sha256=sha256(preview))
            review.write_text(json.dumps(attestation))
            with self.assertRaisesRegex(ReviewError, "shape differs"):
                check_independent_batch(root, BATCH)

    def test_manual_intake_rechecks_bound_model_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            self.assertEqual(check_independent_batch(root, BATCH)["cards"], 96)
            (root / "src/roguard/fixture_eval.py").write_text("# changed fixture\n")
            with self.assertRaisesRegex(ReviewError, "model freeze differs"):
                check_independent_batch(root, BATCH)

    def test_structural_fixture_reaches_adjudicated_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            packets = root / "private_packets"
            build_blind_packets(root, BATCH, "D1", packets)
            rows = load_approved(root, [BATCH])
            by_source = {row["id"]: row for row in rows}
            owner = json.loads((packets / "owner-map.json").read_text())
            provenance = json.loads((root / f"data/synthetic/{BATCH}.provenance.json").read_text())
            factor_by_source = {row["id"]: provenance["pair_factors"][index // 2]
                                for index, row in enumerate(rows)}
            by_opaque = {entry["item_id"]: by_source[entry["source_id"]]
                         for entry in owner["id_map"]}
            answers = []
            for name in ("reviewer-a.jsonl", "reviewer-b.jsonl"):
                packet = [json.loads(line) for line in (packets / name).read_text().splitlines()]
                path = root / f"answers-{name}"
                path.write_text("".join(json.dumps({
                    "item_id": item["item_id"],
                    "card": _fixture_decision(
                        factor_by_source[by_opaque[item["item_id"]]["id"]],
                        bool(by_opaque[item["item_id"]]["labels"])),
                }) + "\n" for item in packet))
                fixture_packet_binding(path, packets / name)
                answers.append(path)
            final = root / "fictional-adjudication.jsonl"
            shutil.copyfile(answers[0], final)
            predictions = root / "fictional-predictions.json"
            predictions.write_text(json.dumps(fixture_blind_predictions(packets, root, "D1")))
            result = evaluate_adjudicated(
                root, packets, answers[0], answers[1], final, predictions)
            registered = evaluate_adjudicated_registered(
                root, packets, answers[0], answers[1], final, predictions)
            self.assertTrue(registered["prediction_model_id_matches_freeze"])
            self.assertFalse(registered["model_execution_attested"])
            self.assertEqual(registered["model_freeze_sha256"], provenance["model_freeze_sha256"])
            saved_predictions = predictions.read_text()
            wrong_model = json.loads(saved_predictions)
            wrong_model["model_id"] = "different_fixture"
            predictions.write_text(json.dumps(wrong_model))
            with self.assertRaisesRegex(ValueError, "Prediction identifier differs"):
                evaluate_adjudicated_registered(
                    root, packets, answers[0], answers[1], final, predictions)
            predictions.write_text(saved_predictions)
            self.assertEqual(result["pairs"]["model_exact_contrast_pairs"], 48)
            self.assertEqual(result["abstract_numeric_target"]["status"], "meets_numeric_target")
            self.assertTrue({"packet_a", "packet_b", "source_batch", "source_preview",
                             "source_attestation", "source_provenance", "taxonomy",
                             "taxonomy_review"} <= set(result["input_sha256"]))
            self.assertEqual([item["adjudicated_contrast_pairs"] for item in result["author_surfaces"]],
                             [24, 24])
            self.assertEqual([item["model_exact_contrast_pairs"] for item in result["author_surfaces"]],
                             [24, 24])
            self.assertLess(result["descriptive_wilson_95"]["recall"][0], 1.0)
            self.assertFalse(result["reviewer_independence_verified"])
            self.assertFalse(result["adjudication_validity_verified"])
            self.assertFalse(result["abstract_numeric_target"]["real_child_language_accuracy_established"])
            result["abstract_numeric_target"]["thresholds"]["minimum_recall"] = 2.0
            repeated = evaluate_adjudicated(
                root, packets, answers[0], answers[1], final, predictions)
            self.assertEqual(repeated["abstract_numeric_target"]["thresholds"]["minimum_recall"], 0.90)
            self.assertEqual(repeated["abstract_numeric_target"]["status"], "meets_numeric_target")
            legacy = root / "legacy-source-ordered.json"
            legacy.write_text(json.dumps({
                "schema_version": 1, "batch": BATCH,
                "batch_sha256": owner["batch_sha256"], "category": "D1",
                "model_id": "legacy_fixture", "predictions": [
                    {"id": row["id"], "predicted": row["labels"], "strict_parse": True}
                    for row in rows]}))
            with self.assertRaisesRegex(ValueError, "exact reviewer packet"):
                evaluate_adjudicated(root, packets, answers[0], answers[1], final, legacy)

            import roguard.human_eval as human_eval_module
            reader = human_eval_module._read_blind_predictions
            for changed_file in (packets / "reviewer-a.jsonl",
                                 root / f"data/synthetic/{BATCH}.jsonl"):
                original_bytes = changed_file.read_bytes()

                def change_after_read(*args, **kwargs):
                    result = reader(*args, **kwargs)
                    changed_file.write_bytes(original_bytes + b"\n")
                    return result

                try:
                    with patch.object(human_eval_module, "_read_blind_predictions",
                                      side_effect=change_after_read):
                        with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                            evaluate_adjudicated(
                                root, packets, answers[0], answers[1], final, predictions)
                finally:
                    changed_file.write_bytes(original_bytes)

            packet_file = packets / "reviewer-b.jsonl"
            original_packet = packet_file.read_bytes()
            target = human_eval_module._numeric_target

            def change_during_report(*args, **kwargs):
                result = target(*args, **kwargs)
                packet_file.write_bytes(original_packet + b"\n")
                return result

            try:
                with patch.object(human_eval_module, "_numeric_target",
                                  side_effect=change_during_report):
                    with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                        evaluate_adjudicated(
                            root, packets, answers[0], answers[1], final, predictions)
            finally:
                packet_file.write_bytes(original_packet)

            opaque_for_source = {entry["source_id"]: entry["item_id"] for entry in owner["id_map"]}
            adjudications = [json.loads(line) for line in final.read_text().splitlines()]
            for item in adjudications:
                if item["item_id"] == opaque_for_source[rows[0]["id"]]:
                    item["card"]["disclosure"]["source_role"] = "minor"
                    item["card"]["disclosure"]["indirect_or_repeated_support_pattern"] = False
                if item["item_id"] == opaque_for_source[rows[2]["id"]]:
                    item["card"]["disclosure"]["safety_or_support_anchor"] = False
            final.write_text("".join(json.dumps(item) + "\n" for item in adjudications))
            invalid_factors = evaluate_adjudicated(
                root, packets, answers[0], answers[1], final, predictions)
            self.assertEqual(invalid_factors["pairs"]["adjudicated_factor_mismatch_pairs"], 1)
            self.assertEqual(invalid_factors["pairs"]["adjudicated_multifact_or_nochange_pairs"], 1)
            self.assertEqual(invalid_factors["pairs"]["adjudicated_valid_one_fact_pairs"], 46)
            additional_mismatches = {opaque_for_source[row["id"]] for row in rows[4:10:2]}
            for item in adjudications:
                if item["item_id"] in additional_mismatches:
                    item["card"]["disclosure"]["source_role"] = "minor"
                    item["card"]["disclosure"]["indirect_or_repeated_support_pattern"] = False
            final.write_text("".join(json.dumps(item) + "\n" for item in adjudications))
            invalid_study = evaluate_adjudicated(
                root, packets, answers[0], answers[1], final, predictions)
            self.assertEqual(invalid_study["pairs"]["adjudicated_valid_one_fact_pairs"], 43)
            self.assertEqual(invalid_study["abstract_numeric_target"]["status"],
                             "inconclusive_insufficient_valid_pairs")
            shutil.copyfile(answers[0], final)

            payload = json.loads(predictions.read_text())
            for row in rows[48:]:
                predicted_row_for_source(payload, owner, row["id"])["predicted"] = []
            predictions.write_text(json.dumps(payload))
            failed = evaluate_adjudicated(root, packets, answers[0], answers[1], final, predictions)
            self.assertEqual(failed["abstract_numeric_target"]["status"], "misses_numeric_target")
            self.assertEqual([item["model_exact_contrast_pairs"] for item in failed["author_surfaces"]],
                             [24, 0])
            self.assertEqual([item["missed_reviews_on_contrast_positives"]
                              for item in failed["author_surfaces"]], [0, 24])

            ambiguous_ids = {opaque_for_source[row["id"]] for row in rows[1:20:2]}
            adjudications = [json.loads(line) for line in final.read_text().splitlines()]
            for item in adjudications:
                if item["item_id"] in ambiguous_ids:
                    item["card"]["disclosure"]["safety_or_support_anchor"] = None
            final.write_text("".join(json.dumps(item) + "\n" for item in adjudications))
            inconclusive = evaluate_adjudicated(
                root, packets, answers[0], answers[1], final, predictions)
            self.assertEqual(inconclusive["pairs"]["adjudicated_contrast_pairs"], 38)
            self.assertEqual(inconclusive["abstract_numeric_target"]["status"],
                             "inconclusive_insufficient_valid_pairs")

    def test_s1_factor_balanced_batch_reaches_one_fact_adjudication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, category="S1")
            packets = root / "private_packets"
            build_blind_packets(root, BATCH, "S1", packets)
            rows = load_approved(root, [BATCH])
            owner = json.loads((packets / "owner-map.json").read_text())
            provenance = json.loads((root / f"data/synthetic/{BATCH}.provenance.json").read_text())
            by_source = {row["id"]: (provenance["pair_factors"][index // 2], bool(row["labels"]))
                         for index, row in enumerate(rows)}
            by_opaque = {entry["item_id"]: by_source[entry["source_id"]]
                         for entry in owner["id_map"]}
            answers = []
            for name in ("reviewer-a.jsonl", "reviewer-b.jsonl"):
                packet = [json.loads(line) for line in (packets / name).read_text().splitlines()]
                path = root / f"answers-{name}"
                path.write_text("".join(json.dumps({
                    "item_id": item["item_id"],
                    "card": _fixture_support_decision(*by_opaque[item["item_id"]]),
                }) + "\n" for item in packet))
                fixture_packet_binding(path, packets / name)
                answers.append(path)
            final = root / "fictional-adjudication.jsonl"
            shutil.copyfile(answers[0], final)
            predictions = root / "fictional-predictions.json"
            predictions.write_text(json.dumps(fixture_blind_predictions(packets, root, "S1")))
            report = evaluate_adjudicated(
                root, packets, answers[0], answers[1], final, predictions)
            self.assertEqual(report["pairs"]["adjudicated_valid_one_fact_pairs"], 48)
            self.assertEqual(report["pairs"]["adjudicated_factor_mismatch_pairs"], 0)
            self.assertEqual(report["abstract_numeric_target"]["status"], "meets_numeric_target")
            self.assertFalse(report["adjudication_validity_verified"])
            payload = json.loads(predictions.read_text())
            first_factor = INDEPENDENT_FACTORS["S1"][0]
            targeted = [index for index, factor in enumerate(provenance["pair_factors"])
                        if factor == first_factor][:4]
            for pair_index in targeted:
                predicted_row_for_source(payload, owner,
                                         rows[2 * pair_index + 1]["id"])["predicted"] = []
            predictions.write_text(json.dumps(payload))
            weak_factor = evaluate_adjudicated(
                root, packets, answers[0], answers[1], final, predictions)
            self.assertEqual(weak_factor["pairs"]["model_exact_valid_one_fact_pairs"], 44)
            self.assertTrue(all(item["exact_valid_one_fact_pair_rate"] >= 0.85
                                for item in weak_factor["author_surfaces"]))
            self.assertEqual(weak_factor["factor_surfaces"][0]["model_exact_valid_one_fact_pairs"], 2)
            self.assertEqual(weak_factor["abstract_numeric_target"]["status"],
                             "misses_numeric_target")

    def test_pending_taxonomy_review_keeps_intake_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            (root / "taxonomy/review_ro_independent.json").unlink()
            with self.assertRaisesRegex(ReviewError, "independent review"):
                load_approved(root, [BATCH])

    def test_bad_author_balance_is_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / f"data/synthetic/{BATCH}.provenance.json"
            provenance = json.loads(path.read_text())
            provenance["pair_authors"][24] = "fixture_author_a"
            path.write_text(json.dumps(provenance))
            review = root / f"data/synthetic/{BATCH}.review.json"
            record = json.loads(review.read_text())
            record["provenance_sha256"] = sha256(path)
            review.write_text(json.dumps(record))
            with self.assertRaisesRegex(ReviewError, "declarations differ"):
                load_approved(root, [BATCH])

    def test_bad_factor_balance_is_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / f"data/synthetic/{BATCH}.provenance.json"
            provenance = json.loads(path.read_text())
            provenance["pair_factors"][0] = "explicit_support_request"
            path.write_text(json.dumps(provenance))
            review = root / f"data/synthetic/{BATCH}.review.json"
            record = json.loads(review.read_text())
            record["provenance_sha256"] = sha256(path)
            review.write_text(json.dumps(record))
            with self.assertRaisesRegex(ReviewError, "factor coverage"):
                load_approved(root, [BATCH])

    def test_review_cannot_precede_authorship(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            path = root / f"data/synthetic/{BATCH}.provenance.json"
            provenance = json.loads(path.read_text())
            provenance["authors"][0]["authored_at"] = "2026-10-01"
            path.write_text(json.dumps(provenance))
            review = root / f"data/synthetic/{BATCH}.review.json"
            record = json.loads(review.read_text())
            record["provenance_sha256"] = sha256(path)
            review.write_text(json.dumps(record))
            with self.assertRaisesRegex(ReviewError, "declarations differ"):
                load_approved(root, [BATCH])

    def test_unreviewed_source_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            data = root / f"data/synthetic/{BATCH}.jsonl"
            data.write_text(data.read_text() + "\n")
            with self.assertRaisesRegex(ReviewError, "digest differs"):
                load_approved(root, [BATCH])

    def test_unadjudicated_author_labels_cannot_be_scored_by_pairs_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            stderr = StringIO()
            with patch.object(sys, "argv", ["roguard-pairs", "--batch", BATCH,
                                            "--predictions", str(root / "unused.json"),
                                            "--root", str(root)]), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as error:
                    pairs_main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("unadjudicated", stderr.getvalue())
            with self.assertRaisesRegex(ValueError, "unadjudicated"):
                evaluate_pairs(load_approved(root, [BATCH]), [
                    {"id": row["id"], "predicted": row["labels"], "strict_parse": True}
                    for row in load_approved(root, [BATCH])])


if __name__ == "__main__":
    unittest.main()
