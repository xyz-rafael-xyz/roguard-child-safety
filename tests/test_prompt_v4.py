import unittest

from roguard.prompt_v4 import applicable_codes, make_prompt_v4, parse_binary


class BinaryPromptTests(unittest.TestCase):
    def test_response_has_two_independent_tasks(self):
        row = {"language": "ro", "source_kind": "response", "text": "fișă simbolică"}
        self.assertEqual(applicable_codes("response"), ("A1", "S1"))
        self.assertIn("Regula pentru A1", make_prompt_v4(row, "A1"))
        self.assertIn("Regula pentru S1", make_prompt_v4(row, "S1"))
        with self.assertRaises(ValueError):
            make_prompt_v4(row, "D1")

    def test_binary_parse_is_strict(self):
        self.assertIs(parse_binary("da\n"), True)
        self.assertIs(parse_binary("nu"), False)
        self.assertIsNone(parse_binary("da, fiindcă"))
        self.assertIsNone(parse_binary("DA"))
        self.assertIsNone(parse_binary("none"))


if __name__ == "__main__":
    unittest.main()
