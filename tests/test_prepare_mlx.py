import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from roguard.review import CATEGORIES, ReviewError, candidate_manifest, taxonomy_sha256

spec = importlib.util.spec_from_file_location(
    "prepare_mlx", Path(__file__).resolve().parents[1] / "training" / "prepare_mlx.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrepareMlxTests(unittest.TestCase):
    def test_v4_export_balances_training_tasks_only(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "binary"
            manifest = module.prepare(root, ["batch-0009", "batch-0010"], "ro", output, "v4")
            train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            dev = [json.loads(line) for line in (output / "valid.jsonl").read_text().splitlines()]
            self.assertEqual((manifest["train_tasks"], manifest["dev_tasks"]), (480, 48))
            self.assertEqual(Counter(row["completion"] for row in train), {"da": 240, "nu": 240})
            self.assertEqual(Counter(row["completion"] for row in dev), {"da": 18, "nu": 30})

    def test_export_requires_review_and_keeps_development_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "taxonomy").mkdir()
            (root / "data" / "synthetic").mkdir(parents=True)
            (root / "taxonomy" / "taxonomy_ro.md").write_text("Approved abstract taxonomy\n", encoding="utf-8")
            (root / "taxonomy" / "review_ro.json").write_text(json.dumps({
                "status": "approved", "reviewer": "fixture", "reviewed_at": "2026-09-29",
                "taxonomy_sha256": taxonomy_sha256(root, "ro"),
            }), encoding="utf-8")
            kinds = {"D1": "message", "R1": "routing_card", "A1": "response",
                     "P1": "permission_card", "G1": "gate_card", "S1": "response"}
            rows = []
            for code in CATEGORIES:
                for split, positive in (("train", True), ("dev", True), ("dev", False)):
                    index = len(rows) + 1
                    rows.append({
                        "id": f"batch-0001-ro-{index:03d}", "language": "ro",
                        "source_kind": kinds[code], "split": split,
                        "text": f"Abstract fixture {code} {split} {index}",
                        "labels": [code] if positive else [], "origin": "abstract_template_v3",
                    })
            batch = root / "data" / "synthetic" / "batch-0001.jsonl"
            batch.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            preview = batch.with_suffix(".preview.md")
            preview.write_text("\n".join(
                f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                for row in rows
            ) + "\n", encoding="utf-8")
            review = batch.with_suffix(".review.json")
            record = candidate_manifest(root, "batch-0001")
            review.write_text(json.dumps(record), encoding="utf-8")
            blocked = root / "blocked"
            with self.assertRaisesRegex(ReviewError, "Pending batch validation"):
                module.prepare(root, ["batch-0001"], "ro", blocked)
            self.assertFalse(blocked.exists())

            record.update(status="approved", reviewer="fixture", reviewed_at="2026-09-29",
                          reviewed_ids=[row["id"] for row in rows])
            review.write_text(json.dumps(record), encoding="utf-8")
            output = root / "prepared"
            manifest = module.prepare(root, ["batch-0001"], "ro", output)
            train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            dev = [json.loads(line) for line in (output / "valid.jsonl").read_text().splitlines()]
            self.assertEqual((manifest["train_rows"], manifest["dev_rows"]), (6, 12))
            self.assertEqual(len(train), 6)
            self.assertEqual(len(dev), 12)
            self.assertEqual({row["completion"] for row in train}, set(CATEGORIES))
            self.assertTrue(all(set(row) == {"prompt", "completion"} for row in train + dev))

            v2_output = root / "prepared_v2"
            v2_manifest = module.prepare(root, ["batch-0001"], "ro", v2_output, "v2")
            v2_train = [json.loads(line) for line in (v2_output / "train.jsonl").read_text().splitlines()]
            self.assertEqual(v2_manifest["prompt_version"], "v2")
            self.assertNotEqual(v2_manifest["train_sha256"], manifest["train_sha256"])
            self.assertTrue(all("nu doar după tipul fișei" in row["prompt"] for row in v2_train))
            self.assertEqual({row["completion"] for row in v2_train}, set(CATEGORIES))


if __name__ == "__main__":
    unittest.main()
