import importlib.util
import unittest
from pathlib import Path

from roguard.prompt import format_codes

spec = importlib.util.spec_from_file_location("run_panel", Path(__file__).resolve().parents[1] / "eval" / "run_panel.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PanelParserTests(unittest.TestCase):
    def test_only_exact_code_output_is_accepted(self):
        self.assertEqual(module.parse_codes("D1,R1"), ("D1", "R1"))
        self.assertEqual(module.parse_codes("NONE"), ())
        self.assertIsNone(module.parse_codes("D1,D1"))
        self.assertIsNone(module.parse_codes("R1,D1"))
        self.assertIsNone(module.parse_codes("D1 because"))
        self.assertEqual(format_codes(["S1", "A1"]), "A1,S1")


if __name__ == "__main__":
    unittest.main()
