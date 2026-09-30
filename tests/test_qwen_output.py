import unittest

from roguard.qwen_output import parse_qwen_binary


class QwenOutputTests(unittest.TestCase):
    def test_lone_decision_and_empty_template_wrapper(self):
        self.assertIs(parse_qwen_binary("da"), True)
        self.assertIs(parse_qwen_binary(" nu\n"), False)
        self.assertIs(parse_qwen_binary("<think>\n\n</think>\n\nda"), True)
        self.assertIs(parse_qwen_binary("<think>\n</think>\nnu"), False)

    def test_unfinished_or_explanatory_output_fails_closed(self):
        for raw in ("<think>because</think>da", "<think>\n", "da because", "da nu", "<think></think>da extra"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_qwen_binary(raw))
