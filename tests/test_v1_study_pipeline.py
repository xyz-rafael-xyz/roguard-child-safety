import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roguard.v1_study_eval import _decision, _pair_factor, evaluate_v1_study
from roguard.v1_study_kit import FACTORS, _schedule, create_v1_author_kit
from roguard.v1_study_packet import prepare_v1_content_review, seal_v1_packets
from roguard.v1_study_seal import create_v1_packet_seal
from roguard.review import sha256


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _facts(category, factor, positive):
    if category == "S1":
        return {"support": {"applicable_fields": [factor],
                            "passed_fields": [] if positive else [factor]}}
    fields = {
        "source_role": "minor", "safety_or_support_anchor": False,
        "indirect_or_repeated_support_pattern": False,
        "explicit_support_request": False, "direct_safety_statement": False,
        "retracted_or_denied": False,
    }
    if factor == "source_role":
        fields.update(source_role="minor" if positive else "adult",
                      safety_or_support_anchor=True,
                      indirect_or_repeated_support_pattern=True)
    elif factor == "safety_or_support_anchor":
        fields.update(safety_or_support_anchor=positive,
                      indirect_or_repeated_support_pattern=True)
    elif factor == "indirect_or_repeated_support_pattern":
        fields.update(safety_or_support_anchor=True,
                      indirect_or_repeated_support_pattern=positive)
    else:
        fields[factor] = positive
    return {"disclosure": fields}


class V1StudyPipelineTests(unittest.TestCase):
    def test_factor_schedules_cover_new_direct_and_conditional_fields(self):
        self.assertIn("direct_safety_statement", FACTORS["D1"])
        self.assertIn("immediate_danger_route", FACTORS["S1"])
        self.assertIn("no_pressing_for_details", FACTORS["S1"])
        for category in FACTORS:
            first, second = _schedule(category, 0), _schedule(category, 1)
            self.assertEqual(len(first), 24)
            self.assertEqual(len(second), 24)
            self.assertEqual(set(first + second), set(FACTORS[category]))

    def test_v1_one_fact_logic_rejects_multiple_changes(self):
        left = _decision("ro", "D1", {"language": "ro", **_facts("D1", "direct_safety_statement", False)})
        right = _decision("ro", "D1", {"language": "ro", **_facts("D1", "direct_safety_statement", True)})
        self.assertEqual(_pair_factor("D1", left, right), "direct_safety_statement")
        changed = dict(right[1], explicit_support_request=True)
        self.assertIsNone(_pair_factor("D1", left, (right[0], changed)))

    def test_private_author_review_packet_and_score_round_trip(self):
        for language, category in (("ro", "D1"), ("uk", "S1")):
            with self.subTest(language=language, category=category), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "review_runs").mkdir()
                frozen = {
                    "path": "eval/prospective/freeze.json", "sha256": "a" * 64,
                    "candidate_taxonomy_sha256": "b" * 64,
                    "model_id": "model_v1_test", "author_pairs_per_cell": 48,
                }
                kit = root / "review_runs/kit"
                candidate = root / "review_runs/candidate"
                sealed = root / "review_runs/sealed"
                with patch("roguard.v1_study_kit.verify_v1_study_freeze", return_value=frozen):
                    create_v1_author_kit(root, language, category,
                                         Path(frozen["path"]), kit)
                for author_id in (f"{language.upper()}-A1", f"{language.upper()}-A2"):
                    path = kit / f"{author_id}.json"
                    book = json.loads(path.read_text(encoding="utf-8"))
                    book["native_language_confirmed_by_author"] = True
                    book["independent_of_model_work_confirmed_by_author"] = True
                    book["authored_at"] = "2026-10-04"
                    for pair in book["pairs"]:
                        number = f"{1 if author_id.endswith('1') else 2} {pair['pair_index']}"
                        if language == "ro":
                            base = "Descriere abstractă a atributelor fictive pentru numărul "
                            pair["negative_abstract_card"] = base + number + " varianta absentă"
                            pair["positive_abstract_card"] = base + number + " varianta prezentă"
                        else:
                            base = "Абстрактний опис вигаданих атрибутів для номера "
                            pair["negative_abstract_card"] = base + number + " відсутній варіант"
                            pair["positive_abstract_card"] = base + number + " наявний варіант"
                    path.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
                with patch("roguard.v1_study_packet.verify_v1_study_freeze", return_value=frozen):
                    result = prepare_v1_content_review(root, kit, candidate)
                    self.assertFalse(result["approved_for_blind_packets"])
                    review = json.loads((candidate / "CONTENT-REVIEW-TEMPLATE.json").read_text(encoding="utf-8"))
                    review["reviewed_at"] = "2026-10-04"
                    for key in review:
                        if key.endswith("_checked"):
                            review[key] = True
                    declaration = root / "review_runs/content-review.json"
                    declaration.write_text(json.dumps(review), encoding="utf-8")
                    seal_v1_packets(root, candidate, declaration, sealed)
                owner = json.loads((sealed / "OWNER-MAP.json").read_text(encoding="utf-8"))
                seal_path = root / "eval/prospective/packet-seal.json"
                with patch("roguard.v1_study_seal.verify_v1_study_freeze", return_value=frozen):
                    create_v1_packet_seal(root, sealed, seal_path)
                expected = {}
                for pair in owner["pairs"]:
                    for positive, key in ((False, "negative_item_id"), (True, "positive_item_id")):
                        expected[pair[key]] = {"item_id": pair[key], "card": {
                            "language": language, **_facts(category, pair["factor"], positive)}}
                for slot in ("a", "b"):
                    packet = [json.loads(line) for line in
                              (sealed / f"reviewer-{slot}.jsonl").read_text(encoding="utf-8").splitlines()]
                    _jsonl(root / f"review_runs/answers-{slot}.jsonl",
                           [expected[row["item_id"]] for row in packet])
                _jsonl(root / "review_runs/adjudicated.jsonl", list(expected.values()))
                packet_a = [json.loads(line) for line in
                            (sealed / "reviewer-a.jsonl").read_text(encoding="utf-8").splitlines()]
                positives = {pair["positive_item_id"] for pair in owner["pairs"]}
                prediction = {
                    "schema_version": 2,
                    "packet_sha256": owner["packet_sha256"]["reviewer-a"],
                    "category": category, "language": language,
                    "model_id": frozen["model_id"],
                    "predictions": [{"item_id": row["item_id"],
                                     "predicted": [category] if row["item_id"] in positives else [],
                                     "strict_parse": True} for row in packet_a],
                }
                prediction_path = root / "review_runs/predictions.json"
                prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
                with patch("roguard.v1_study_eval.verify_v1_study_freeze", return_value=frozen), \
                     patch("roguard.v1_study_packet.verify_v1_study_freeze", return_value=frozen):
                    report = evaluate_v1_study(
                        root, sealed, seal_path, root / "review_runs/answers-a.jsonl",
                        root / "review_runs/answers-b.jsonl",
                        root / "review_runs/adjudicated.jsonl", prediction_path)
                self.assertEqual(report["status"], "meets_numeric_target")
                self.assertEqual(report["adjudicated_valid_one_fact_pairs"], 48)
                self.assertEqual(report["model_exact_valid_one_fact_pairs"], 48)
                self.assertEqual(report["annotator_decision_disagreements"], 0)
                owner["pairs"][0]["negative_item_id"], owner["pairs"][1]["negative_item_id"] = (
                    owner["pairs"][1]["negative_item_id"], owner["pairs"][0]["negative_item_id"])
                (sealed / "OWNER-MAP.json").write_text(json.dumps(owner), encoding="utf-8")
                seal_record = json.loads(seal_path.read_text(encoding="utf-8"))
                seal_record["owner_map_sha256"] = sha256(sealed / "OWNER-MAP.json")
                seal_path.write_text(json.dumps(seal_record), encoding="utf-8")
                with patch("roguard.v1_study_eval.verify_v1_study_freeze", return_value=frozen), \
                     patch("roguard.v1_study_packet.verify_v1_study_freeze", return_value=frozen):
                    with self.assertRaisesRegex(ValueError, "mapping differs"):
                        evaluate_v1_study(
                            root, sealed, seal_path, root / "review_runs/answers-a.jsonl",
                            root / "review_runs/answers-b.jsonl",
                            root / "review_runs/adjudicated.jsonl", prediction_path)


if __name__ == "__main__":
    unittest.main()
