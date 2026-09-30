import json
import runpy
import unittest
from collections import Counter
from pathlib import Path

from roguard.cli import assess_json
from roguard.review import ReviewError, load_approved, load_train_dev, sha256
from training.generate_hybrid_holdout import build_contracts, build_rows

ROOT = Path(__file__).resolve().parents[1]


class HybridStudyTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "models/ro-mmbert-v11-abstract/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_saved_mixed_input_result_recomputes(self):
        verify = runpy.run_path(str(ROOT / "eval/verify_mmbert_v14.py"))["verify"]
        report = verify(ROOT)
        self.assertTrue(report["preregistered_success"])
        self.assertEqual(report["advisory_exact_pairs"], 24)
        self.assertEqual(report["declared_contract_exact_pairs"], 48)

    def test_registered_test_and_privileged_sidecar_reproduce(self):
        rows = load_approved(ROOT, ["batch-0026"])
        sides = [json.loads(line) for line in
                 (ROOT / "data/synthetic/batch-0026.contracts.jsonl").read_text(
                     encoding="utf-8").splitlines()]
        audit = json.loads((ROOT / "data/synthetic/batch-0026.audit.json").read_text(
            encoding="utf-8"))
        self.assertEqual(rows, build_rows())
        self.assertEqual(sides, build_contracts())
        self.assertEqual(len(rows), 144)
        self.assertEqual(audit["contract_sidecar_sha256"],
                         sha256(ROOT / "data/synthetic/batch-0026.contracts.jsonl"))
        self.assertEqual(Counter(code for row in rows for code in row["labels"]),
                         {code: 12 for code in ("D1", "R1", "A1", "P1", "G1", "S1")})
        self.assertEqual(sum(side["contract"] is None for side in sides), 48)
        for row, side in zip(rows, sides):
            self.assertEqual(row["id"], side["id"])
            if side["contract"] is not None:
                result = assess_json({"language": "ro", **side["contract"]})
                self.assertEqual([finding["category"] for finding in result["findings"]
                                  if finding["status"] != "pass"], row["labels"])
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0021", "batch-0022", "batch-0026"], "ro")


if __name__ == "__main__":
    unittest.main()
