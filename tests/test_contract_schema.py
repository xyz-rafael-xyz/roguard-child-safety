import json
import unittest
from pathlib import Path

from roguard.cli import CARD_FIELDS, EVENT_FIELDS, OPTIONAL_CARD_FIELDS, assess_json
from roguard.assess import DECISION_CODES
from roguard.policy import SUPPORT_FIELDS
from roguard.review import CATEGORIES, KINDS

ROOT = Path(__file__).resolve().parents[1]


class ContractSchemaTests(unittest.TestCase):
    def test_editor_schema_tracks_cli_fields(self):
        schema = json.loads((ROOT / "schema" / "contract-input.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(set(schema["properties"]), {"language", "proposed_use", *CARD_FIELDS})
        self.assertEqual(schema["required"], ["language"])
        self.assertEqual(
            {tuple(branch["required"]) for branch in schema["anyOf"]},
            {(name,) for name in CARD_FIELDS} | {("proposed_use",)},
        )
        for name, fields in CARD_FIELDS.items():
            with self.subTest(card=name):
                definition = schema["$defs"][name]
                self.assertEqual(set(definition["properties"]), fields)
                self.assertEqual(set(definition["required"]), fields - OPTIONAL_CARD_FIELDS.get(name, set()))
                self.assertFalse(definition["additionalProperties"])
        event = schema["$defs"]["event"]
        self.assertEqual(set(event["properties"]), EVENT_FIELDS)
        self.assertEqual(set(event["required"]), EVENT_FIELDS - {"expires_at"})
        support_values = schema["$defs"]["nullableSupportSet"]["anyOf"][1]["items"]["enum"]
        self.assertEqual(set(support_values), SUPPORT_FIELDS)

    def test_documented_examples_still_pass_cli(self):
        report_schema = json.loads((ROOT / "schema" / "contract-report.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(report_schema["properties"]["schema_version"]["const"], 2)
        self.assertEqual(report_schema["properties"]["caller_facts_verified"]["const"], False)
        self.assertEqual(report_schema["properties"]["language_verified"]["const"], False)
        finding = report_schema["$defs"]["finding"]["properties"]
        self.assertEqual(set(finding["category"]["enum"]), set(CATEGORIES))
        self.assertEqual(set(finding["source_kind"]["enum"]), set(KINDS))
        self.assertEqual(set(finding["decision_code"]["enum"]),
                         {value for statuses in DECISION_CODES.values() for value in statuses.values()})
        for name in ("routing_ro", "routing_uk", "contracts_ro", "contracts_uk"):
            with self.subTest(example=name):
                payload = json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))
                report = assess_json(payload)
                self.assertEqual(report["language"], payload["language"])
                self.assertEqual(set(report), set(report_schema["properties"]) - {"proposed_use"})
                self.assertTrue(report["findings"])
                self.assertEqual(set(report["checked_categories"]),
                                 {item["category"] for item in report["findings"]})
                for item in report["findings"]:
                    self.assertEqual(set(item), set(finding))


if __name__ == "__main__":
    unittest.main()
