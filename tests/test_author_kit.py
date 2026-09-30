"""Mechanical author-kit fixtures do not represent real independent reviewers."""

import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest

from roguard.author_kit import assemble_candidate, new_author_kit as _new_author_kit, seal_candidate
from roguard.review import CATEGORIES, ReviewError, load_approved, sha256, taxonomy_sha256
from roguard.study_freeze import verify_study_freeze

SOURCE = Path(__file__).resolve().parents[1]
BATCH = "batch-0098"
FREEZE = Path("eval/prospective/fixture-freeze.json")


def new_author_kit(root, batch, language, category, output_dir):
    return _new_author_kit(root, batch, language, category, output_dir,
                           freeze_record=FREEZE)


def _root(path: Path, language: str, *, approved: bool = True) -> Path:
    root = path / "repo"
    (root / "taxonomy").mkdir(parents=True)
    (root / "data/synthetic").mkdir(parents=True)
    (root / "review_runs").mkdir()
    shutil.copyfile(SOURCE / f"taxonomy/taxonomy_{language}.md",
                    root / f"taxonomy/taxonomy_{language}.md")
    digest = taxonomy_sha256(root, language)
    review = {
        "status": "approved" if approved else "pending", "taxonomy_sha256": digest,
        "reviews": ([{"kind": kind, "reviewer": reviewer,
                     "reviewed_at": "2026-09-30", "independent_of_author": True,
                     "summary": "Mechanical fixture only",
                     "category_decisions": {code: "accept" for code in CATEGORIES}}
                    for kind, reviewer in (("language", "fixture_language"),
                                           ("child_safety", "fixture_safety"))]
                    if approved else []),
    }
    name = "review_ro_independent.json" if language == "ro" else "review_uk.json"
    (root / "taxonomy" / name).write_text(json.dumps(review))
    adapter = root / "models/fixture-abstract"
    adapter.mkdir(parents=True)
    weight = adapter / "adapter_model.safetensors"
    weight.write_bytes(b"fixture weight, not a model")
    manifest = adapter / "research.json"
    manifest.write_text(json.dumps({"trained_language": language,
                                    "shared_threshold": 0.5,
                                    "adapter_weight_sha256": sha256(weight)}))
    source = root / "src/roguard"
    source.mkdir(parents=True)
    predictor = source / "fixture_predict.py"
    evaluator = source / "fixture_eval.py"
    predictor.write_text("# mechanical fixture\n")
    evaluator.write_text("# mechanical fixture\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                    "commit", "-qm", "fixture model"], cwd=root, check=True)
    model_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           text=True).strip()
    path = root / FREEZE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "status": "registered_baseline_before_independent_batch",
        "registered_on": "2026-09-30",
        "pre_author_git_commit": model_commit,
        "language": language, "categories": ["D1", "S1"],
        "input_scope": "independently_authored_abstract_cards_only",
        "adapter_dir": "models/fixture-abstract", "model_id": "fixture", "cutoff": 0.5,
        "prediction_model_id": "fixture_only",
        "predictor_entrypoint": "src/roguard/fixture_predict.py",
        "evaluator_entrypoint": "src/roguard/fixture_eval.py",
        "sha256": {str(file.relative_to(root)): sha256(file)
                   for file in (weight, manifest, predictor, evaluator)},
    }))
    subprocess.run(["git", "add", str(FREEZE)], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                    "commit", "-qm", "fixture freeze"], cwd=root, check=True)
    return root


def _fill(path: Path) -> None:
    book = json.loads(path.read_text())
    book.update(native_language=book["language"], independent_of_project=True,
                authored_at="2026-09-30")
    for pair in book["pairs"]:
        if book["language"] == "uk":
            base = (f"Умовна таблиця {book['author_id']}, пара {pair['pair_index']:02d}; "
                    f"показник {pair['factor']}")
            pair["negative_abstract_card"] = base + " відсутній; інші поля незмінні."
            pair["positive_abstract_card"] = base + " наявний; інші поля незмінні."
        else:
            base = (f"Tabel abstract {book['author_id']} perechea {pair['pair_index']:02d}; "
                    f"indicatorul {pair['factor']} este")
            pair["negative_abstract_card"] = base + " absent; celelalte rubrici sunt constante."
            pair["positive_abstract_card"] = base + " prezent; celelalte rubrici sunt constante."
    path.write_text(json.dumps(book, ensure_ascii=False, indent=2) + "\n")


def _declaration(candidate: Path) -> Path:
    declaration = candidate.parent / "content-review.json"
    template = json.loads((candidate / "content-review-template.json").read_text())
    self_declared = {**template, "reviewer_id": "fixture_reviewer",
                     "reviewed_at": "2026-09-30",
                     "abstract_only_checked": True,
                     "no_realistic_content_checked": True,
                     "pair_logic_checked": True,
                     "factor_balance_checked": True,
                     "prior_overlap_checked": True}
    declaration.write_text(json.dumps(self_declared))
    return declaration


class AuthorKitTests(unittest.TestCase):
    def test_historical_v16_record_cannot_start_a_new_study(self):
        with self.assertRaisesRegex(ValueError, "omits the named predictor_entrypoint"):
            verify_study_freeze(SOURCE,
                                Path("eval/prospective/ro-v16-baseline-freeze.json"),
                                "ro", "D1")

    def test_romanian_and_ukrainian_structural_intake(self):
        for language, category in (("ro", "D1"), ("uk", "S1")):
            with self.subTest(language=language, category=category), tempfile.TemporaryDirectory() as directory:
                root = _root(Path(directory), language)
                kit = root / "review_runs" / "kit"
                created = new_author_kit(root, BATCH, language, category, kit)
                self.assertFalse(created["human_authorship_or_independence_verified"])
                self.assertEqual(stat.S_IMODE(kit.stat().st_mode), 0o700)
                workbooks = [kit / "author-a.json", kit / "author-b.json"]
                for book in workbooks:
                    self.assertEqual(stat.S_IMODE(book.stat().st_mode), 0o600)
                    self.assertIsNone(json.loads(book.read_text())["authored_at"])
                    _fill(book)
                candidate = root / "review_runs" / "candidate"
                staged = assemble_candidate(root, *workbooks,
                                            kit / "owner-assignments.json", candidate)
                self.assertEqual((staged["cards"], staged["pairs"]), (96, 48))
                self.assertFalse(staged["approved_for_repository"])
                self.assertFalse((root / f"data/synthetic/{BATCH}.jsonl").exists())
                pending = json.loads((candidate / f"{BATCH}.review.json").read_text())
                self.assertEqual(pending["status"], "pending")
                declaration = _declaration(candidate)
                result = seal_candidate(root, candidate, declaration)
                self.assertEqual(result["cards"], 96)
                self.assertFalse(result["actual_author_independence_verified_by_software"])
                self.assertFalse(result["intended_labels_adjudicated"])
                rows = load_approved(root, [BATCH])
                self.assertEqual(len(rows), 96)
                self.assertEqual({row["source_kind"] for row in rows},
                                 {"message" if category == "D1" else "response"})
                self.assertEqual([row["labels"] for row in rows[:2]], [[], [category]])
                provenance = json.loads((root / f"data/synthetic/{BATCH}.provenance.json").read_text())
                self.assertEqual(len(provenance["pair_factors"]), 48)
                self.assertEqual(len(set(provenance["pair_authors"])), 2)

    def test_pending_taxonomy_blocks_new_workbooks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro", approved=False)
            with self.assertRaisesRegex(ReviewError, "Pending independent"):
                new_author_kit(root, BATCH, "ro", "D1", root / "review_runs/kit")
            self.assertFalse((root / "review_runs/kit").exists())

    def test_model_freeze_must_match_before_authors_receive_workbooks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            (root / "models/fixture-abstract/adapter_model.safetensors").write_bytes(
                b"changed fixture weight")
            with self.assertRaisesRegex(ValueError, "Study freeze file differs"):
                new_author_kit(root, BATCH, "ro", "D1", root / "review_runs/kit")
            self.assertFalse((root / "review_runs/kit").exists())

    def test_uncommitted_freeze_record_blocks_author_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            freeze = root / FREEZE
            freeze.write_text(freeze.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "not committed at HEAD"):
                new_author_kit(root, BATCH, "ro", "D1", root / "review_runs/kit")
            self.assertFalse((root / "review_runs/kit").exists())

    def test_freeze_rejects_files_changed_after_claimed_source_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            predictor = root / "src/roguard/fixture_predict.py"
            predictor.write_text("# later fixture predictor\n")
            freeze = root / FREEZE
            record = json.loads(freeze.read_text())
            record["sha256"]["src/roguard/fixture_predict.py"] = sha256(predictor)
            freeze.write_text(json.dumps(record))
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                            "commit", "-qm", "later code and amended freeze"], cwd=root, check=True)
            with self.assertRaisesRegex(ValueError, "lack registered bytes at source commit"):
                new_author_kit(root, BATCH, "ro", "D1", root / "review_runs/kit")
            self.assertFalse((root / "review_runs/kit").exists())

    def test_freeze_names_prediction_and_evaluation_entrypoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            freeze = root / FREEZE
            record = json.loads(freeze.read_text())
            del record["evaluator_entrypoint"]
            freeze.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "omits the named evaluator_entrypoint"):
                new_author_kit(root, BATCH, "ro", "D1", root / "review_runs/kit")

    def test_changed_freeze_blocks_assembly_and_sealing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            kit = root / "review_runs/kit"
            new_author_kit(root, BATCH, "ro", "D1", kit)
            workbooks = [kit / "author-a.json", kit / "author-b.json"]
            for book in workbooks:
                _fill(book)
            code = root / "src/roguard/fixture_predict.py"
            code.write_text("# changed fixture\n")
            with self.assertRaisesRegex(ValueError, "Study freeze file differs"):
                assemble_candidate(root, *workbooks, kit / "owner-assignments.json",
                                   root / "review_runs/candidate")
            code.write_text("# mechanical fixture\n")
            candidate = root / "review_runs/candidate"
            assemble_candidate(root, *workbooks, kit / "owner-assignments.json", candidate)
            declaration = _declaration(candidate)
            code.write_text("# changed fixture again\n")
            with self.assertRaisesRegex(ValueError, "Study freeze file differs"):
                seal_candidate(root, candidate, declaration)
            self.assertFalse((root / f"data/synthetic/{BATCH}.jsonl").exists())

    def test_candidate_change_or_missing_review_blocks_repository_intake(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            kit = root / "review_runs/kit"
            new_author_kit(root, BATCH, "ro", "D1", kit)
            workbooks = [kit / "author-a.json", kit / "author-b.json"]
            for book in workbooks:
                _fill(book)
            candidate = root / "review_runs/candidate"
            assemble_candidate(root, *workbooks, kit / "owner-assignments.json", candidate)
            with self.assertRaisesRegex(ValueError, "ordinary file"):
                seal_candidate(root, candidate, root / "review_runs/missing-review.json")
            declaration = _declaration(candidate)
            declared = json.loads(declaration.read_text())
            declared["reviewer_id"] = json.loads(workbooks[0].read_text())["author_id"]
            declaration.write_text(json.dumps(declared))
            with self.assertRaisesRegex(ValueError, "Candidate authors"):
                seal_candidate(root, candidate, declaration)
            preview = candidate / f"{BATCH}.preview.md"
            preview.write_text(preview.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed after assembly"):
                seal_candidate(root, candidate, declaration)
            self.assertFalse((root / f"data/synthetic/{BATCH}.jsonl").exists())

    def test_failed_final_batch_validation_removes_new_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            kit = root / "review_runs/kit"
            new_author_kit(root, BATCH, "ro", "D1", kit)
            workbooks = [kit / "author-a.json", kit / "author-b.json"]
            for book in workbooks:
                _fill(book)
            candidate = root / "review_runs/candidate"
            assemble_candidate(root, *workbooks, kit / "owner-assignments.json", candidate)
            declaration = _declaration(candidate)
            first = json.loads((candidate / f"{BATCH}.jsonl").read_text().splitlines()[0])
            prior = root / "data/synthetic/batch-0097.jsonl"
            prior.write_text(json.dumps({**first, "id": "batch-0097-ro-00001"}) + "\n")
            with self.assertRaisesRegex(ValueError, "overlap audit changed"):
                seal_candidate(root, candidate, declaration)
            self.assertFalse(any((root / "data/synthetic" / f"{BATCH}{suffix}").exists()
                                 for suffix in (".jsonl", ".preview.md", ".provenance.json",
                                                ".review.json")))

    def test_unreviewed_candidate_cannot_be_written_into_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            kit = root / "review_runs/kit"
            with self.assertRaisesRegex(ValueError, "outside Git"):
                new_author_kit(root, BATCH, "ro", "D1", root / "data/synthetic/kit")
            new_author_kit(root, BATCH, "ro", "D1", kit)
            workbooks = [kit / "author-a.json", kit / "author-b.json"]
            for book in workbooks:
                _fill(book)
            with self.assertRaisesRegex(ValueError, "outside Git"):
                assemble_candidate(root, *workbooks, kit / "owner-assignments.json",
                                   root / "data/synthetic/candidate")

    def test_changed_factor_order_is_rejected_even_when_balance_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            kit = root / "review_runs/kit"
            new_author_kit(root, BATCH, "ro", "D1", kit)
            workbooks = [kit / "author-a.json", kit / "author-b.json"]
            for book in workbooks:
                _fill(book)
            changed = json.loads(workbooks[0].read_text())
            pairs = changed["pairs"]
            other = next(index for index in range(1, len(pairs))
                         if pairs[index]["factor"] != pairs[0]["factor"])
            pairs[0]["factor"], pairs[other]["factor"] = (
                pairs[other]["factor"], pairs[0]["factor"])
            workbooks[0].write_text(json.dumps(changed, ensure_ascii=False))
            with self.assertRaisesRegex(ValueError, "changed an assigned factor"):
                assemble_candidate(root, *workbooks, kit / "owner-assignments.json",
                                   root / "review_runs/candidate")

    def test_wrong_script_is_rejected_before_content_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "uk")
            kit = root / "review_runs/kit"
            new_author_kit(root, BATCH, "uk", "S1", kit)
            workbooks = [kit / "author-a.json", kit / "author-b.json"]
            for book in workbooks:
                _fill(book)
            altered = json.loads(workbooks[0].read_text())
            altered["pairs"][0]["negative_abstract_card"] = (
                "Tabel abstract cu rubrici simbolice, toate câmpurile rămân constante.")
            workbooks[0].write_text(json.dumps(altered, ensure_ascii=False))
            target = root / "review_runs/candidate"
            with self.assertRaisesRegex(ReviewError, "language script"):
                assemble_candidate(root, *workbooks, kit / "owner-assignments.json", target)
            self.assertFalse(target.exists())

    def test_prior_overlap_is_bound_to_separate_review_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _root(Path(directory), "ro")
            kit = root / "review_runs/kit"
            new_author_kit(root, BATCH, "ro", "D1", kit)
            workbooks = [kit / "author-a.json", kit / "author-b.json"]
            for book in workbooks:
                _fill(book)
            first = json.loads(workbooks[0].read_text())["pairs"][0]["negative_abstract_card"]
            prior = root / "data/synthetic/batch-0097.jsonl"
            prior.write_text(json.dumps({"language": "ro", "source_kind": "message",
                                         "text": first.replace("abstract", "simbolic")},
                                        ensure_ascii=False) + "\n")
            candidate = root / "review_runs/candidate"
            assemble_candidate(root, *workbooks, kit / "owner-assignments.json", candidate)
            manifest = json.loads((candidate / "assembly-manifest.json").read_text())
            self.assertGreaterEqual(manifest["prior_overlap_audit"]["near_overlap_cards"], 1)
            declaration = _declaration(candidate)
            incomplete = json.loads(declaration.read_text())
            incomplete["prior_overlap_checked"] = False
            declaration.write_text(json.dumps(incomplete))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                seal_candidate(root, candidate, declaration)
            incomplete["prior_overlap_checked"] = True
            declaration.write_text(json.dumps(incomplete))
            self.assertEqual(seal_candidate(root, candidate, declaration)["cards"], 96)


if __name__ == "__main__":
    unittest.main()
