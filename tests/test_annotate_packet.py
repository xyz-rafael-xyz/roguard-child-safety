"""Packet annotation resumes without copying IDs or storing abstract text."""

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from roguard.annotate_packet import annotate_packet


def _packet(path: Path, category: str = "D1", language: str = "ro") -> None:
    rows = [
        {"item_id": f"item_{number:02d}", "language": language, "category": category,
         "source_kind": "message" if category == "D1" else "response",
         "abstract_card": f"Fișă simbolică {number}; numai valori inventate."}
        for number in (1, 2)
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _reader(answers):
    stream = iter(answers)
    return lambda: next(stream, "")


class AnnotatePacketTests(unittest.TestCase):
    def test_resumes_only_missing_rows_and_keeps_text_out_of_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, answers = Path(directory) / "packet.jsonl", Path(directory) / "answers.jsonl"
            _packet(packet)
            shown = []
            with self.assertRaisesRegex(ValueError, "incompletă"):
                annotate_packet(packet, answers, _reader(["1\n", "1\n", "2\n", "1\n"]), shown.append)
            saved = answers.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0])["item_id"], "item_01")
            self.assertEqual(stat.S_IMODE(answers.stat().st_mode), 0o600)
            self.assertIn("Fișă simbolică 1", "\n".join(shown))
            result = annotate_packet(packet, answers,
                                     _reader(["2\n", "2\n", "2\n", "2\n"]), shown.append)
            self.assertEqual(result["resumed_from"], 1)
            self.assertEqual(result["rows_completed"], 2)
            self.assertEqual([json.loads(line)["item_id"] for line in answers.read_text().splitlines()],
                             ["item_01", "item_02"])
            self.assertNotIn("Fișă simbolică", answers.read_text())
            self.assertEqual(annotate_packet(packet, answers, _reader([]), shown.append)["resumed_from"], 2)

    def test_rejects_packet_with_reference_labels_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, answers = Path(directory) / "packet.jsonl", Path(directory) / "answers.jsonl"
            _packet(packet)
            rows = [json.loads(line) for line in packet.read_text().splitlines()]
            rows[0]["labels"] = ["D1"]
            packet.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "invalid fields"):
                annotate_packet(packet, answers, _reader([]), lambda _: None)
            self.assertFalse(answers.exists())

    def test_rejects_wrong_existing_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, answers = Path(directory) / "packet.jsonl", Path(directory) / "answers.jsonl"
            _packet(packet)
            bad = {"item_id": "item_02", "card": {"language": "ro", "disclosure": {
                "source_role": "minor", "safety_or_support_anchor": True,
                "indirect_or_repeated_support_pattern": False,
                "explicit_support_request": False}}}
            answers.write_text(json.dumps(bad) + "\n")
            os.chmod(answers, 0o600)
            with self.assertRaisesRegex(ValueError, "packet prefix"):
                annotate_packet(packet, answers, _reader([]), lambda _: None)
            self.assertEqual(len(answers.read_text().splitlines()), 1)

    def test_support_packet_writes_numeric_judgments_only(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, answers = Path(directory) / "packet.jsonl", Path(directory) / "answers.jsonl"
            _packet(packet, "S1")
            result = annotate_packet(packet, answers, _reader(["2\n"] * 16), lambda _: None)
            self.assertEqual(result["rows_completed"], 2)
            cards = [json.loads(line)["card"] for line in answers.read_text().splitlines()]
            self.assertTrue(all(set(card) == {"language", "support"} for card in cards))
            self.assertTrue(all(card["support"]["applicable_fields"] == [] for card in cards))

    def test_hard_link_to_packet_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, answers = Path(directory) / "packet.jsonl", Path(directory) / "answers.jsonl"
            _packet(packet)
            os.link(packet, answers)
            with self.assertRaisesRegex(ValueError, "must differ"):
                annotate_packet(packet, answers, _reader([]), lambda _: None)
            self.assertEqual(len(packet.read_text().splitlines()), 2)

    def test_ukrainian_packet_keeps_language_and_opaque_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, answers = Path(directory) / "packet.jsonl", Path(directory) / "answers.jsonl"
            _packet(packet, language="uk")
            result = annotate_packet(packet, answers, _reader(["4\n", "3\n", "3\n", "3\n"] * 2),
                                     lambda _: None)
            self.assertEqual(result["language"], "uk")
            cards = [json.loads(line) for line in answers.read_text().splitlines()]
            self.assertEqual([item["item_id"] for item in cards], ["item_01", "item_02"])
            self.assertTrue(all(item["card"]["language"] == "uk" for item in cards))
            self.assertTrue(all(item["card"]["disclosure"]["source_role"] is None
                                for item in cards))


if __name__ == "__main__":
    unittest.main()
