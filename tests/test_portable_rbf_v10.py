"""Portable inference checks for a hash-bound v10 frozen-feature head."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import pytest

np = pytest.importorskip('numpy')

from roguard._portable_rbf_v10 import PortableRBF


class PortableRBFTests(unittest.TestCase):
    def test_separates_two_opposite_features_and_rejects_bad_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            vectors = np.zeros((2, 2560), dtype=np.float64)
            vectors[0, 0] = 1
            vectors[1, 1] = 1
            asset = Path(directory) / 'model.npz'
            np.savez_compressed(asset, support_vectors=vectors,
                                dual_coef=np.array([[-1., 1.]], dtype=np.float64),
                                intercept=np.array([0.], dtype=np.float64),
                                gamma=np.array([1.], dtype=np.float64))
            model = PortableRBF(asset, expected_sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
            first = np.zeros(2560, dtype=np.float32)
            second = np.zeros(2560, dtype=np.float32)
            first[0], second[1] = 1, 1
            self.assertLess(model.margin(first), 0)
            self.assertGreater(model.margin(second), 0)
            with self.assertRaisesRegex(ValueError, 'RBF asset bytes differ'):
                PortableRBF(asset, expected_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'Expected finite float32'):
                model.margin(first.astype(np.float64))

    def test_joint_feature_normalizes_each_source_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            vectors = np.zeros((2, 3584), dtype=np.float64)
            vectors[0, 0], vectors[0, 2560] = 2 ** -.5, 2 ** -.5
            vectors[1, 1], vectors[1, 2561] = 2 ** -.5, 2 ** -.5
            asset = Path(directory) / 'joint.npz'
            np.savez_compressed(asset, support_vectors=vectors,
                                dual_coef=np.array([[-1., 1.]], dtype=np.float64),
                                intercept=np.array([0.], dtype=np.float64),
                                gamma=np.array([1.], dtype=np.float64))
            model = PortableRBF(asset, expected_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                                feature_mode='joint')
            first = np.zeros(3584, dtype=np.float32)
            second = np.zeros(3584, dtype=np.float32)
            first[0], first[2560] = 10, 1
            second[1], second[2561] = 1, 10
            self.assertLess(model.margin(first), 0)
            self.assertGreater(model.margin(second), 0)


if __name__ == '__main__':
    unittest.main()
