"""Keep the editor schema's primitive constraints aligned with the CLI."""

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from roguard.cli import assess_json

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = tuple(
    ROOT / "examples" / f"{name}_{language}.json"
    for language in ("ro", "uk")
    for name in ("contracts", "routing", "proposed_use", "reviewer_evidence")
    if (ROOT / "examples" / f"{name}_{language}.json").exists()
)
MUTATIONS = (None, "", " ", [], {}, 0, -1, True, False, "x", ["x"])


def leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaf_paths(child, prefix + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaf_paths(child, prefix + (index,))
    else:
        yield prefix


def changed(original, path, replacement):
    payload = copy.deepcopy(original)
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement
    return payload


class EditorSchemaAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads((ROOT / "schema/contract-input.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        cls.validator = Draft202012Validator(schema)

    def test_cli_accepted_leaf_mutations_are_schema_valid(self):
        checked = 0
        for path in EXAMPLES:
            original = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(self.validator.is_valid(original), path)
            for leaf in leaf_paths(original):
                for replacement in MUTATIONS:
                    payload = changed(original, leaf, replacement)
                    try:
                        assess_json(payload)
                    except (ValueError, TypeError):
                        continue
                    checked += 1
                    self.assertTrue(
                        self.validator.is_valid(payload),
                        f"Editor rejects CLI-accepted input: {path.name} {leaf} {replacement!r}",
                    )
        self.assertGreater(checked, 100)

    def test_editor_rejects_cli_invalid_primitive_values(self):
        original = json.loads((ROOT / "examples/contracts_ro.json").read_text(encoding="utf-8"))
        cases = (
            (("readability", "max_words"), -1),
            (("permission", "at"), -1),
            (("permission", "events", 0, "sequence"), -1),
            (("routing", "recipient_roles", 0), " "),
            (("routing", "allowed_by_scope", 0, "principal"), ""),
        )
        for path, replacement in cases:
            with self.subTest(path=path):
                payload = changed(original, path, replacement)
                self.assertFalse(self.validator.is_valid(payload))
                with self.assertRaises(ValueError):
                    assess_json(payload)


if __name__ == "__main__":
    unittest.main()
