"""Synthetic numeric streams exercise the future adjudicated evaluation path."""

import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from roguard.blind_packets import build_blind_packets
from roguard.human_eval import _one_fact_change, evaluate_adjudicated
from roguard.human_eval_bound import evaluate_adjudicated_bound, main as public_eval_main
from roguard.packet_binding import binding_path
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
BATCH = "batch-0035"


def _card(positive: bool, *, unknown: bool = False) -> dict:
    return {"language": "ro", "disclosure": {
        "source_role": "minor",
        "safety_or_support_anchor": None if unknown else positive,
        "indirect_or_repeated_support_pattern": positive,
        "explicit_support_request": False,
    }}


def _prepare(directory: Path, batch: str = BATCH, category: str = "D1") -> tuple[Path, Path, Path, Path, Path, list[dict]]:
    packets = directory / "packets"
    build_blind_packets(ROOT, batch, category, packets)
    rows = [row for row in load_approved(ROOT, [batch])
            if row["source_kind"] == ("message" if category == "D1" else "response")]
    owner = json.loads((packets / "owner-map.json").read_text())
    by_id = {row["id"]: row for row in rows}
    source_for_item = {item["item_id"]: by_id[item["source_id"]] for item in owner["id_map"]}
    paths = []
    for suffix in ("a", "b", "final"):
        path = directory / f"{suffix}.jsonl"
        packet_name = "reviewer-b.jsonl" if suffix == "b" else "reviewer-a.jsonl"
        packet = [json.loads(line) for line in (packets / packet_name).read_text().splitlines()]
        path.write_text("".join(json.dumps({
            "item_id": item["item_id"],
            "card": (_card(category in source_for_item[item["item_id"]]["labels"])
                     if category == "D1" else {"language": "ro", "support": {
                         "applicable_fields": ["acknowledgement"],
                         "passed_fields": ([] if category in source_for_item[item["item_id"]]["labels"]
                                           else ["acknowledgement"]),
                     }}),
        }) + "\n" for item in packet))
        paths.append(path)
    predictions = directory / "predictions.json"
    payload = {"schema_version": 1, "batch": batch,
               "batch_sha256": sha256(ROOT / f"data/synthetic/{batch}.jsonl"),
               "category": category, "model_id": "test_frozen",
               "predictions": [{"id": row["id"],
                                "predicted": [category] if category in row["labels"] else [],
                                "strict_parse": True} for row in rows]}
    predictions.write_text(json.dumps(payload))
    return packets, *paths, predictions, rows


class HumanEvalTests(unittest.TestCase):
    def test_public_evaluator_hashes_both_packet_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, a, b, final, predictions, _ = _prepare(Path(directory))
            for answer, packet_name in ((a, "reviewer-a.jsonl"),
                                        (b, "reviewer-b.jsonl")):
                binding = binding_path(answer)
                binding.write_text(json.dumps({"schema_version": 1,
                                               "packet_sha256": sha256(packets / packet_name)}))
                binding.chmod(0o600)
            report = evaluate_adjudicated_bound(ROOT, packets, a, b, final, predictions)
            self.assertTrue(report["annotation_packet_binding_verified"])
            self.assertEqual(report["input_sha256"]["reviewer_a_packet_binding"],
                             sha256(binding_path(a)))
            self.assertEqual(report["input_sha256"]["reviewer_b_packet_binding"],
                             sha256(binding_path(b)))
            self.assertNotIn('"abstract_card":', json.dumps(report))

    def test_public_evaluator_rejects_binding_change_during_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, a, b, final, predictions, _ = _prepare(Path(directory))
            for answer, packet_name in ((a, "reviewer-a.jsonl"),
                                        (b, "reviewer-b.jsonl")):
                binding = binding_path(answer)
                binding.write_text(json.dumps({"schema_version": 1,
                                               "packet_sha256": sha256(packets / packet_name)}))
                binding.chmod(0o600)
            def change_after_scoring(*args):
                report = evaluate_adjudicated(*args)
                binding_path(a).write_text(binding_path(a).read_text() + " ")
                return report
            with patch("roguard.human_eval_bound.evaluate_adjudicated",
                       side_effect=change_after_scoring):
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    evaluate_adjudicated_bound(ROOT, packets, a, b, final, predictions)

    def test_public_evaluator_marks_unbound_demonstration(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, a, b, final, predictions, _ = _prepare(Path(directory))
            report = evaluate_adjudicated_bound(ROOT, packets, a, b, final, predictions)
            self.assertFalse(report["annotation_packet_binding_verified"])
            self.assertNotIn("reviewer_a_packet_binding", report["input_sha256"])

    def test_public_cli_writes_private_report_once(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, a, b, final, predictions, _ = _prepare(Path(directory))
            output = Path(directory) / "report.json"
            arguments = ["roguard-human-eval", str(packets), str(a), str(b),
                         str(final), str(predictions), "--root", str(ROOT),
                         "--output", str(output)]
            with patch("sys.argv", arguments):
                public_eval_main()
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
                self.assertFalse(json.loads(output.read_text())[
                    "annotation_packet_binding_verified"])
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    public_eval_main()

    def test_support_pair_requires_one_declared_failed_field(self):
        fields = ["acknowledgement", "no_blame"]
        left = {"support": {"applicable_fields": fields, "passed_fields": fields}}
        right = {"support": {"applicable_fields": fields,
                             "passed_fields": ["acknowledgement"]}}
        self.assertEqual(_one_fact_change("S1", left, right), "no_blame")
        right["support"]["passed_fields"] = []
        self.assertIsNone(_one_fact_change("S1", left, right))
        right["support"]["passed_fields"] = ["acknowledgement"]
        right["support"]["applicable_fields"] = ["no_blame"]
        self.assertIsNone(_one_fact_change("S1", left, right))

    def test_perfect_mechanical_stream_has_no_human_validity_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, a, b, final, predictions, _ = _prepare(Path(directory))
            report = evaluate_adjudicated(ROOT, packets, a, b, final, predictions)
            self.assertEqual(report["complete_adjudications"], 48)
            self.assertEqual(report["adjudicated_positive"], 24)
            self.assertEqual(report["model_counts_on_complete"]["tp"], 24)
            self.assertEqual(report["model_counts_on_complete"]["tn"], 24)
            self.assertEqual(report["pairs"]["model_exact_contrast_pairs"], 24)
            self.assertEqual(report["always_review"]["fp"], 24)
            self.assertFalse(report["reviewer_independence_verified"])
            self.assertFalse(report["adjudication_validity_verified"])
            self.assertFalse(report["model_frozen_before_adjudication_verified"])
            self.assertEqual(report["abstract_numeric_target"]["status"],
                             "not_applicable_generator_batch")
            self.assertNotIn('"abstract_card":', json.dumps(report))

    def test_ambiguity_and_parse_failure_stay_out_of_confusion_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, a, b, final, predictions, rows = _prepare(Path(directory))
            owner = json.loads((packets / "owner-map.json").read_text())
            source_to_item = {entry["source_id"]: entry["item_id"] for entry in owner["id_map"]}
            adjudications = [json.loads(line) for line in final.read_text().splitlines()]
            for item in adjudications:
                if item["item_id"] == source_to_item[rows[1]["id"]]:
                    item["card"] = _card(True, unknown=True)
            final.write_text("".join(json.dumps(item) + "\n" for item in adjudications))
            payload = json.loads(predictions.read_text())
            payload["predictions"][2]["predicted"] = ["D1"]
            payload["predictions"][3]["predicted"] = []
            payload["predictions"][3]["strict_parse"] = False
            predictions.write_text(json.dumps(payload))
            report = evaluate_adjudicated(ROOT, packets, a, b, final, predictions)
            self.assertEqual(report["ambiguous_adjudications"], 1)
            self.assertEqual(report["complete_adjudications"], 47)
            self.assertEqual(report["model_counts_on_complete"]["fp"], 1)
            self.assertEqual(report["model_counts_on_complete"]["positive_abstentions"], 1)
            self.assertEqual(report["model_parsed_cards"], 47)
            self.assertEqual(report["pairs"]["incomplete_pairs"], 1)
            self.assertEqual(report["pairs"]["adjudicated_contrast_pairs"], 23)
            self.assertEqual(report["pairs"]["model_exact_contrast_pairs"], 22)

    def test_prediction_file_cannot_carry_generator_labels_or_wrong_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, a, b, final, predictions, _ = _prepare(Path(directory))
            payload = json.loads(predictions.read_text())
            payload["predictions"][0]["expected"] = []
            predictions.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "output contract"):
                evaluate_adjudicated(ROOT, packets, a, b, final, predictions)
            del payload["predictions"][0]["expected"]
            payload["batch_sha256"] = "0" * 64
            predictions.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "exact abstract batch"):
                evaluate_adjudicated(ROOT, packets, a, b, final, predictions)

    def test_s1_response_fields_are_scored_against_adjudicated_support_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, a, b, final, predictions, _ = _prepare(
                Path(directory), batch="batch-0034", category="S1")
            report = evaluate_adjudicated(ROOT, packets, a, b, final, predictions)
            self.assertEqual(report["category"], "S1")
            self.assertEqual(report["adjudicated_positive"], 24)
            self.assertEqual(report["model_counts_on_complete"]["tp"], 24)
            self.assertEqual(report["pairs"]["model_exact_contrast_pairs"], 24)


if __name__ == "__main__":
    unittest.main()
