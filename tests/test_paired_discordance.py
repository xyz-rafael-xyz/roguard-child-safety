import unittest


from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from compare_qwen_balanced_v8 import paired_discordance


class PairedDiscordanceTests(unittest.TestCase):
    def test_exact_two_sided_p_value(self):
        candidate = (True, True, True, False, False)
        comparator = (False, False, False, True, False)
        result = paired_discordance(candidate, comparator)
        self.assertEqual(result["candidate_only_exact_pairs"], 3)
        self.assertEqual(result["comparator_only_exact_pairs"], 1)
        self.assertEqual(result["two_sided_exact_mcnemar_p"], 0.625)

    def test_tied_pairs(self):
        result = paired_discordance((True, False), (True, False))
        self.assertEqual(result["two_sided_exact_mcnemar_p"], 1.0)
