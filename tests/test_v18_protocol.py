"""V18 factor targets and held-out batch coverage."""

import unittest
from pathlib import Path

from roguard.prompt_v18 import FIELDS, decision, make_factor_prompt
from roguard.review import load_approved
from training.v18_facts import load_d1_rows_with_facts

ROOT = Path(__file__).resolve().parents[1]


class V18ProtocolTests(unittest.TestCase):
    def test_factor_rule_requires_minor_anchor_and_support_route(self):
        values = {field: False for field in FIELDS}
        self.assertFalse(decision(values))
        values.update(minor_source=True, support_anchor=True, indirect_pattern=True)
        self.assertTrue(decision(values))
        values["indirect_pattern"] = False
        self.assertFalse(decision(values))
        values["explicit_request"] = True
        self.assertTrue(decision(values))
        values["support_anchor"] = False
        self.assertFalse(decision(values))

    def test_prompt_is_one_field_at_a_time(self):
        prompts = [make_factor_prompt("Fișă de probă simbolică.", field) for field in FIELDS]
        self.assertEqual(len(set(prompts)), 4)
        with self.assertRaises(ValueError):
            make_factor_prompt("", "minor_source")

    def test_registered_factors_reproduce_all_d1_labels(self):
        expected = {"batch-0028": 192, "batch-0031": 48, "batch-0032": 48,
                    "batch-0033": 48, "batch-0034": 96}
        for batch, count in expected.items():
            with self.subTest(batch=batch):
                cards = load_d1_rows_with_facts(ROOT, batch)
                self.assertEqual(len(cards), count)
                self.assertEqual(sum(bool(row["labels"]) for row, _ in cards), count // 2)

    def test_new_test_has_two_distinct_surfaces_and_s1_coverage(self):
        dev = load_approved(ROOT, ["batch-0033"])
        test = load_approved(ROOT, ["batch-0034"])
        self.assertEqual((len(dev), len(test)), (96, 144))
        self.assertFalse({row["text"] for row in dev} & {row["text"] for row in test})
        self.assertEqual(sum(row["labels"] == ["D1"] for row in test), 48)
        self.assertEqual(sum(row["labels"] == ["S1"] for row in test), 24)


if __name__ == "__main__":
    unittest.main()
