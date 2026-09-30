import unittest

from roguard import count_words
from roguard.cli import assess_json


class WordCountTests(unittest.TestCase):
    def test_unicode_count_convention_for_neutral_ro_and_uk_text(self):
        self.assertEqual(count_words("Ana are 2 cărți; s-a întors."), 6)
        self.assertEqual(count_words("Книга лежить на столі; пів’яблука."), 5)
        self.assertEqual(count_words("ca\u0302rte și cântec"), 3)
        self.assertEqual(count_words("  ...  "), 0)
        self.assertEqual(count_words("a--b a_b a‑b"), 5)

    def test_cli_counts_text_and_rejects_wrong_supplied_measurement(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                card = {"declared_age": 10, "measured_words": None, "max_words": 3,
                        "text": "Ana are 2 cărți" if language == "ro" else "Книга лежить на столі"}
                finding = assess_json({"language": language, "readability": card})["findings"][0]
                self.assertEqual(finding["reason_codes"], ["WORD_CAP_EXCEEDED"])
                self.assertNotIn("text", finding)
                card["measured_words"] = 4
                self.assertEqual(assess_json({"language": language, "readability": card})["findings"][0]["status"], "review")
                card["measured_words"] = 3
                with self.assertRaisesRegex(ValueError, "differs from local count"):
                    assess_json({"language": language, "readability": card})

    def test_invalid_text_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Response text must be a string"):
            count_words(None)
        with self.assertRaisesRegex(ValueError, "Response text must be a string"):
            assess_json({"language": "ro", "readability": {
                "declared_age": 10, "measured_words": None, "max_words": 3, "text": 7}})


if __name__ == "__main__":
    unittest.main()
