"""Lexical overlap is a review flag, never evidence of semantic independence."""

import json
from pathlib import Path
import tempfile
import unittest

from roguard.intake_overlap import audit_prior_overlap


class IntakeOverlapTests(unittest.TestCase):
    def test_cross_pair_copy_is_flagged_without_echoing_cards(self):
        base = ("Fișa simbolică descrie rubrici abstracte, ordonate în tabel; "
                "toate câmpurile rămân constante la verificare.")
        rows = [{"language": "ro", "source_kind": "message", "text": value}
                for value in (base + " Indicatorul A lipsește.",
                              base + " Indicatorul A este prezent.",
                              base + " Indicatorul B lipsește.",
                              base + " Indicatorul B este prezent.")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/synthetic").mkdir(parents=True)
            report = audit_prior_overlap(root, rows, "batch-0098")
        cross_pair = report["within_candidate_cross_pair"]
        self.assertEqual(cross_pair["cross_pair_comparisons"], 4)
        self.assertEqual(cross_pair["near_overlap_cards"], 4)
        self.assertEqual(report["prior_cards_compared"], 0)
        self.assertNotIn(base, json.dumps(report, ensure_ascii=False))
        self.assertFalse(cross_pair["semantic_independence_verified"])

    def test_pair_mate_is_excluded_from_cross_pair_warning(self):
        rows = [{"language": "ro", "source_kind": "message", "text": value}
                for value in (
                    "Tabelul albastru conține marcaje despre rubrici și secvențe ordonate; starea este absentă.",
                    "Tabelul albastru conține marcaje despre rubrici și secvențe ordonate; starea este prezentă.",
                    "Diagrama verde inventată prezintă forme circulare și numerale fără niciun tabel; forma lipsește.",
                    "Diagrama verde inventată prezintă forme circulare și numerale fără niciun tabel; forma apare.",
                )]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/synthetic").mkdir(parents=True)
            report = audit_prior_overlap(root, rows, "batch-0098")
        cross_pair = report["within_candidate_cross_pair"]
        self.assertEqual(cross_pair["near_overlap_cards"], 0)
        self.assertEqual(cross_pair["cross_pair_comparisons"], 4)

    def test_near_copy_is_flagged_without_returning_text(self):
        prior = ("Fișa simbolică descrie rubrici abstracte, ordonate în tabel; "
                 "toate câmpurile rămân constante la verificare.")
        candidate = prior.replace("ordonate", "aranjate")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/synthetic"
            source.mkdir(parents=True)
            (source / "batch-0097.jsonl").write_text(json.dumps({
                "language": "ro", "source_kind": "message", "text": prior,
            }, ensure_ascii=False) + "\n")
            report = audit_prior_overlap(root, [{"language": "ro", "source_kind": "message",
                                                 "text": candidate}], "batch-0098")
            self.assertEqual(report["near_overlap_cards"], 1)
            self.assertEqual(report["flagged"][0]["prior_batch"], "batch-0097")
            self.assertNotIn(prior, json.dumps(report, ensure_ascii=False))
            self.assertNotIn(candidate, json.dumps(report, ensure_ascii=False))
            self.assertFalse(report["semantic_independence_verified"])

    def test_other_language_and_kind_do_not_enter_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/synthetic"
            source.mkdir(parents=True)
            (source / "batch-0097.jsonl").write_text(json.dumps({
                "language": "uk", "source_kind": "response",
                "text": "Умовна таблиця містить лише абстрактні позначки."},
                ensure_ascii=False) + "\n")
            report = audit_prior_overlap(root, [{"language": "ro", "source_kind": "message",
                                                 "text": "Fișă abstractă cu rubrici constante."}],
                                         "batch-0098")
            self.assertEqual(report["prior_cards_compared"], 0)
            self.assertIsNone(report["maximum_similarity"])
            before = report["prior_batch_sha256"]["batch-0097"]
            (source / "batch-0097.jsonl").write_text(json.dumps({
                "language": "uk", "source_kind": "response",
                "text": "Інша умовна таблиця містить лише абстрактні позначки."},
                ensure_ascii=False) + "\n")
            changed = audit_prior_overlap(root, [{"language": "ro", "source_kind": "message",
                                                  "text": "Fișă abstractă cu rubrici constante."}],
                                          "batch-0098")
            self.assertEqual(changed["prior_cards_compared"], 0)
            self.assertNotEqual(changed["prior_batch_sha256"]["batch-0097"], before)


if __name__ == "__main__":
    unittest.main()
