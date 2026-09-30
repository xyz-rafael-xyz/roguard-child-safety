import unittest

from roguard.token_margin import choice_score, choice_token_ids


class TokenMarginTests(unittest.TestCase):
    def test_two_choice_score_is_bounded_and_symmetric(self):
        self.assertEqual(choice_score(0, 0), 0.5)
        self.assertAlmostEqual(choice_score(1, 0), 1 - choice_score(0, 1))
        self.assertGreater(choice_score(1000, -1000), 0.999)
        self.assertLess(choice_score(-1000, 1000), 0.001)

    def test_answer_tokens_are_checked_against_chat_prefix(self):
        class Tokenizer:
            eos_token_id = 2

            def apply_chat_template(self, messages, **kwargs):
                return [10, 11, 20 if messages[-1]["content"] == "da" else 21, 2]

        messages = [{"role": "user", "content": "simbol"}]
        self.assertEqual(choice_token_ids(Tokenizer(), messages, [10, 11]), (20, 21))
        with self.assertRaisesRegex(ValueError, "one token"):
            choice_token_ids(Tokenizer(), messages, [99])


if __name__ == "__main__":
    unittest.main()
