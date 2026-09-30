import json
import unittest
from pathlib import Path

from roguard.contrast import contrast_declared_contracts


ROOT = Path(__file__).resolve().parents[1]


class ContrastTests(unittest.TestCase):
    def test_one_fact_route_change_is_explained_without_input_values(self):
        before = json.loads((ROOT / "examples/contrast_route_before.json").read_text())
        after = json.loads((ROOT / "examples/contrast_route_after.json").read_text())
        report = contrast_declared_contracts(before, after)
        self.assertEqual(report["changed_path"], "/routing/proposed_recipient")
        self.assertFalse(report["before_review_suggested"])
        self.assertTrue(report["after_review_suggested"])
        self.assertEqual(report["transitions"][0]["after_reason_codes"], ["PRINCIPAL_SCOPE_CONFLICT"])
        self.assertNotIn("element-fictiv", json.dumps(report))
        self.assertNotIn("tutore", json.dumps(report))

    def test_two_changes_are_rejected(self):
        before = json.loads((ROOT / "examples/contrast_route_before.json").read_text())
        after = json.loads((ROOT / "examples/contrast_route_after.json").read_text())
        after["routing"]["purpose"] = "B"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            contrast_declared_contracts(before, after)

    def test_nested_shape_change_is_rejected(self):
        before = json.loads((ROOT / "examples/contrast_route_before.json").read_text())
        after = json.loads((ROOT / "examples/contrast_route_after.json").read_text())
        extra = after["routing"]["allowed_by_scope"][0].copy()
        extra["item"] = "alt-element"
        after["routing"]["allowed_by_scope"].append(extra)
        with self.assertRaisesRegex(ValueError, "same length"):
            contrast_declared_contracts(before, after)
