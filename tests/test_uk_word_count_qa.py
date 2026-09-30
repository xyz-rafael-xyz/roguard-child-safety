"""Regression for the frozen Ukrainian A1 probe's numeral-agreement defect."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from uk_word_count_qa import audit_card, audit_file, word_form  # noqa: E402


class UkrainianWordCountQaTests(unittest.TestCase):
    def test_cardinal_forms_and_fixed_contexts(self):
        cases = {0: "слів", 1: "слово", 2: "слова", 4: "слова", 5: "слів",
                 11: "слів", 12: "слів", 14: "слів", 15: "слів",
                 21: "слово", 22: "слова", 100: "слів", 101: "слово",
                 111: "слів", 112: "слів", 124: "слова"}
        for number, expected in cases.items():
            with self.subTest(number=number):
                self.assertEqual(word_form(number), expected)
                self.assertEqual(audit_card(f"межа — {number} {expected}; виміряно {number} {expected}."), ())
        with self.assertRaises(ValueError):
            word_form(-1)
        self.assertEqual(audit_card("межа — 62 слів; виміряно 63 слів."), (
            {"number": 62, "actual": "слів", "expected": "слова"},
            {"number": 63, "actual": "слів", "expected": "слова"},
        ))

    def test_historical_defect_is_detected_without_changing_frozen_data(self):
        frozen = ROOT / "data/synthetic/uk-contrast-0001.jsonl"
        report = audit_file(frozen)
        self.assertEqual((report["cards"], report["frames"], report["errors"]), (8, 16, 7))
        self.assertEqual({item["id"] for item in report["problems"]}, {
            "uk-contrast-0001-09-0", "uk-contrast-0001-09-1",
            "uk-contrast-0001-11-1", "uk-contrast-0001-12-0",
            "uk-contrast-0001-12-1",
        })
        replacements = {54: "слова", 62: "слова", 63: "слова", 92: "слова"}
        rows = [json.loads(line) for line in frozen.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["source_kind"] == "response":
                for number, expected in replacements.items():
                    row["text"] = row["text"].replace(f"{number} слів", f"{number} {expected}")
        with tempfile.TemporaryDirectory() as temp:
            corrected = Path(temp) / "corrected.jsonl"
            corrected.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                                 encoding="utf-8")
            clean = audit_file(corrected)
        self.assertEqual((clean["cards"], clean["frames"], clean["errors"]), (8, 16, 0))


if __name__ == "__main__":
    unittest.main()
