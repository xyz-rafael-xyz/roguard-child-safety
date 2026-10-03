"""Content-free decision sensitivity checks."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from roguard.explore import explore_declared_contract


ROOT = Path(__file__).resolve().parents[1]


class ExploreTests(unittest.TestCase):
    def test_finds_routing_flip_without_copying_identifiers(self):
        payload = json.loads((ROOT / "examples/contrast_route_before.json").read_text())
        report = explore_declared_contract(payload)
        route_changes = [row for row in report["decision_changes"]
                         if row["path"] == "/routing/proposed_recipient"]
        self.assertTrue(route_changes)
        self.assertEqual(route_changes[0]["changes"][0]["after_status"], "review")
        self.assertNotIn("element-fictiv", json.dumps(report))
        self.assertFalse(report["caller_facts_verified"])

    def test_finds_permission_expiry_and_support_field(self):
        permission = {"principal": "minor", "item": "K", "purpose": "A",
                      "recipient": "reviewer", "at": 3, "proposed_use": True,
                      "use_required": True,
                      "events": [{"sequence": 1, "principal": "minor", "item": "K",
                                  "purpose": "A", "recipient": "reviewer", "action": "grant",
                                  "expires_at": 4}]}
        support = {"applicable_fields": ["next_step"], "passed_fields": ["next_step"]}
        report = explore_declared_contract({"language": "uk", "permission": permission,
                                            "support": support})
        self.assertTrue(any(row["path"] == "/permission/at" and
                            any(change["after_reason_codes"] == ["UNAUTHORIZED_REUSE"]
                                for change in row["changes"])
                            for row in report["decision_changes"]))
        self.assertTrue(any(row.get("field_code") == "next_step" and
                            row["mutation"] == "remove"
                            for row in report["decision_changes"]))
        self.assertFalse(report["truncated"])

    def test_limit_and_bad_inputs_fail_closed(self):
        payload = json.loads((ROOT / "examples/contracts_ro.json").read_text())
        report = explore_declared_contract(payload, limit=1)
        self.assertEqual(report["tested"] + report["skipped_invalid"], 1)
        self.assertTrue(report["truncated"])
        with self.assertRaisesRegex(ValueError, "limit"):
            explore_declared_contract(payload, limit=True)
        with self.assertRaisesRegex(ValueError, "fields differ"):
            explore_declared_contract({"language": "ro", "gate": {"unknown": True}})

    def test_cli_runs_from_stdin_without_reflecting_payload_values(self):
        payload = (ROOT / "examples/contrast_route_before.json").read_text()
        run = subprocess.run([sys.executable, "-m", "roguard.explore", "--limit", "32"],
                             input=payload, capture_output=True, text=True, check=True)
        report = json.loads(run.stdout)
        self.assertTrue(report["decision_changes"])
        self.assertNotIn("element-fictiv", run.stdout)


if __name__ == "__main__":
    unittest.main()
