import json
import subprocess
import sys
import unittest
from pathlib import Path

from roguard.cli import assess_json, assess_jsonl
from roguard.contrast import contrast_declared_contracts


ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_reviewer_evidence_templates_and_one_fact_d1_contrasts(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                template = json.loads((ROOT / "examples" / f"reviewer_evidence_{language}.json").read_text(
                    encoding="utf-8"))
                report = assess_json(template)
                self.assertEqual(report["checked_categories"], ["D1"])
                self.assertEqual(report["findings"][0]["status"], "incomplete")
                self.assertEqual(report["findings"][0]["reason_codes"], ["MISSING_SIGNAL_EVIDENCE"])
                before = json.loads((ROOT / "examples" / f"contrast_disclosure_before_{language}.json").read_text(
                    encoding="utf-8"))
                after = json.loads((ROOT / "examples" / f"contrast_disclosure_after_{language}.json").read_text(
                    encoding="utf-8"))
                change = contrast_declared_contracts(before, after)
                self.assertEqual(change["changed_path"], "/disclosure/safety_or_support_anchor")
                self.assertEqual(change["transitions"][0]["before_status"], "pass")
                self.assertEqual(change["transitions"][0]["after_status"], "review")
                self.assertFalse(change["before_review_suggested"])
                self.assertTrue(change["after_review_suggested"])

    def test_example_reports_independent_findings_in_both_languages(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                path = ROOT / "examples" / f"routing_{language}.json"
                result = subprocess.run([sys.executable, "-m", "roguard", str(path)],
                                        capture_output=True, text=True, check=True)
                report = json.loads(result.stdout)
                self.assertEqual(report["language"], language)
                self.assertEqual(report["schema_version"], 2)
                self.assertEqual(report["checked_categories"], ["R1", "G1"])
                self.assertFalse(report["caller_facts_verified"])
                self.assertFalse(report["language_verified"])
                self.assertTrue(report["review_suggested"])
                self.assertEqual([(item["category"], item["status"]) for item in report["findings"]],
                                 [("R1", "review"), ("G1", "pass")])
                self.assertEqual(report["findings"][0]["reason_codes"], ["PRINCIPAL_SCOPE_CONFLICT"])
                self.assertEqual(report["findings"][0]["evidence_basis"], "declared_policy")
                self.assertEqual(report["findings"][0]["decision_code"], "REVIEW_ROUTE")
                self.assertEqual(report["findings"][1]["decision_code"], "NO_GATE_SIGNAL")
                self.assertEqual(report["findings"][1]["evidence_basis"], "declared_workflow")

    def test_unknown_fields_and_wrong_types_fail_closed(self):
        payload = json.loads((ROOT / "examples" / "routing_ro.json").read_text(encoding="utf-8"))
        payload["routing"]["extra"] = "ignored?"
        with self.assertRaisesRegex(ValueError, "fields differ"):
            assess_json(payload)
        del payload["routing"]["extra"]
        payload["gate"]["human_approval_required"] = "yes"
        with self.assertRaisesRegex(ValueError, "booleans"):
            assess_json(payload)

    def test_empty_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            assess_json({"language": "ro"})

    def test_permission_requires_explicit_utility_contract(self):
        permission = {
            "principal": "minor", "item": "symbol", "purpose": "A", "recipient": "reviewer",
            "at": 3, "proposed_use": False,
            "events": [{"sequence": 1, "principal": "minor", "item": "symbol",
                        "purpose": "A", "recipient": "reviewer", "action": "grant"}],
        }
        with self.assertRaisesRegex(ValueError, "fields differ"):
            assess_json({"language": "ro", "permission": permission})
        permission["use_required"] = False
        self.assertEqual(assess_json({"language": "ro", "permission": permission})["findings"][0]["status"], "pass")
        permission["use_required"] = True
        finding = assess_json({"language": "ro", "permission": permission})["findings"][0]
        self.assertEqual(finding["reason_codes"], ["OVER_WITHHOLD"])

    def test_narrowed_permission_is_not_revived_by_resume_in_either_language(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                events = [{"sequence": index, "principal": "minor", "item": "K",
                           "purpose": "A", "recipient": "reviewer", "action": action}
                          for index, action in enumerate(("grant", "narrow", "resume"), 1)]
                permission = {"principal": "minor", "item": "K", "purpose": "A",
                              "recipient": "reviewer", "at": 4, "proposed_use": True,
                              "use_required": True, "events": events}
                result = assess_json({"language": language, "permission": permission})
                self.assertEqual(result["findings"][0]["reason_codes"],
                                 ["UNAUTHORIZED_REUSE"])

    def test_jsonl_batch_is_ordered_and_validated_before_output(self):
        ro = json.loads((ROOT / "examples" / "routing_ro.json").read_text(encoding="utf-8"))
        uk = json.loads((ROOT / "examples" / "routing_uk.json").read_text(encoding="utf-8"))
        raw = "\n".join(json.dumps(item, ensure_ascii=False) for item in (ro, uk)) + "\n"
        reports = assess_jsonl(raw)
        self.assertEqual([report["language"] for report in reports], ["ro", "uk"])
        run = subprocess.run([sys.executable, "-m", "roguard", "--jsonl"], input=raw,
                             capture_output=True, text=True, check=True)
        self.assertEqual([json.loads(line)["language"] for line in run.stdout.splitlines()], ["ro", "uk"])
        invalid = raw + "{}\n"
        failed = subprocess.run([sys.executable, "-m", "roguard", "--jsonl"], input=invalid,
                                capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(failed.stdout, "")
        self.assertIn("JSONL line 3", failed.stderr)

    def test_json_input_rejects_duplicate_keys_and_nonstandard_constants(self):
        cases = (
            ('{"language":"ro","language":"uk","gate":{}}', "Duplicate JSON object key"),
            ('{"language":"ro","readability":{"declared_age":NaN}}', "Nonstandard JSON numeric constant"),
            ('{"language":"uk","readability":{"declared_age":Infinity}}', "Nonstandard JSON numeric constant"),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                run = subprocess.run([sys.executable, "-m", "roguard"], input=raw,
                                     capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(run.stdout, "")
                self.assertIn(expected, run.stderr)

    def test_jsonl_rejects_nested_duplicate_without_partial_output(self):
        valid = (ROOT / "examples/contracts_ro.json").read_text(encoding="utf-8")
        invalid = '{"language":"uk","gate":{"review_owner":"a","review_owner":"b"}}'
        run = subprocess.run([sys.executable, "-m", "roguard", "--jsonl"],
                             input=valid.replace("\n", "") + "\n" + invalid + "\n",
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(run.stdout, "")
        self.assertIn("JSONL line 2: Duplicate JSON object key", run.stderr)

    def test_full_examples_include_declared_evidence_for_all_six_categories(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                payload = json.loads((ROOT / "examples" / f"contracts_{language}.json").read_text(encoding="utf-8"))
                report = assess_json(payload)
                self.assertEqual({item["category"] for item in report["findings"]},
                                 {"D1", "R1", "A1", "P1", "G1", "S1"})
                disclosure = next(item for item in report["findings"] if item["category"] == "D1")
                support = next(item for item in report["findings"] if item["category"] == "S1")
                self.assertEqual(disclosure["reason_codes"], ["DECLARED_INDIRECT_SUPPORT"])
                self.assertEqual(support["reason_codes"], ["SUPPORT_NEXT_STEP_FAILED"])

    def test_boundary_reason_separates_recurrence_from_new_unlisted_field(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                source = json.loads((ROOT / "examples" / f"contracts_{language}.json").read_text(
                    encoding="utf-8"))
                boundary = source["boundary"]
                protected = boundary["protected_fields"][0]
                boundary["required_fields"] = []
                boundary["proposed_fields"] = ["unlisted"]
                result = assess_json({"language": language, "boundary": boundary})
                self.assertEqual(result["findings"][0]["reason_codes"], ["UNAUTHORIZED_REUSE"])
                boundary["proposed_fields"] = [protected, "unlisted"]
                result = assess_json({"language": language, "boundary": boundary})
                self.assertEqual(result["findings"][0]["reason_codes"],
                                 ["BOUNDARY_RECURRENCE", "UNAUTHORIZED_REUSE"])


if __name__ == "__main__":
    unittest.main()
