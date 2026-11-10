"""Bounded NumPy inference for frozen Gemma4 v10 RBF heads."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path


class PortableRBF:
    """Predict one signed margin from frozen final or joint features."""

    def __init__(self, path: str | Path, *, expected_sha256: str, feature_mode='final'):
        import numpy as np

        if feature_mode not in ('final', 'joint'):
            raise ValueError('Expected final or joint feature mode')
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
            char not in '0123456789abcdef' for char in expected_sha256
        ):
            raise ValueError('Expected lowercase SHA-256 required')
        with Path(path).open('rb') as source:
            raw = source.read(20 * 1024 * 1024 + 1)
        if len(raw) > 20 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('RBF asset bytes differ')
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != {'support_vectors', 'dual_coef', 'intercept', 'gamma'}:
                raise ValueError('RBF asset inventory differs')
            vectors = archive['support_vectors']
            coefficients = archive['dual_coef']
            intercept = archive['intercept']
            gamma = archive['gamma']
        count = len(vectors)
        width = 2560 if feature_mode == 'final' else 3584
        if (vectors.dtype != np.float64 or vectors.ndim != 2 or vectors.shape[1:] != (width,)
                or not 1 <= count <= 2048 or coefficients.dtype != np.float64
                or coefficients.shape != (1, count) or intercept.dtype != np.float64
                or intercept.shape != (1,) or gamma.dtype != np.float64 or gamma.shape != (1,)
                or not all(np.isfinite(value).all() for value in (vectors, coefficients, intercept, gamma))
                or gamma[0] <= 0):
            raise ValueError('Invalid finite RBF tensors')
        self._vectors = vectors.copy()
        self._coefficients = coefficients.reshape(-1).copy()
        self._intercept = float(intercept[0])
        self._gamma = float(gamma[0])
        self._vector_norms = np.sum(self._vectors * self._vectors, axis=1)
        self.feature_mode = feature_mode
        self.width = width
        for value in (self._vectors, self._coefficients, self._vector_norms):
            value.flags.writeable = False
        self.identity = expected_sha256

    def margin(self, feature) -> float:
        import numpy as np

        value = np.asarray(feature)
        if value.dtype != np.float32 or value.shape != (self.width,) or not np.isfinite(value).all():
            raise ValueError('Expected finite float32 feature with bound width')
        normalized = value.copy()
        if self.feature_mode == 'joint':
            for block in (normalized[:2560], normalized[2560:]):
                block /= np.maximum(np.linalg.norm(block), np.float32(1e-12))
            normalized /= np.sqrt(np.float32(2))
        else:
            normalized /= np.maximum(np.linalg.norm(normalized), np.float32(1e-12))
        x = normalized.astype(np.float64)
        distance = np.maximum(np.sum(x * x) + self._vector_norms - 2 * self._vectors @ x, 0)
        result = float(np.exp(-self._gamma * distance) @ self._coefficients + self._intercept)
        if not np.isfinite(result):
            raise ValueError('Nonfinite RBF margin')
        return result
