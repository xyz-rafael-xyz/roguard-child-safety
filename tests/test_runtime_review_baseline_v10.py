"""Reference rebuilding rejects numerical drift, malformed arrays and nonfinite values."""
import pytest

from training.extract_runtime_review_baseline_v10 import bounded_error


def test_reference_tolerances_remain_absolute():
    np = pytest.importorskip('numpy')
    left = np.array([1000., 0.])
    assert bounded_error(left, left.copy(), 1e-3) == 0.
    assert bounded_error(left, np.array([1000., .0009]), 1e-3) == .0009
    with pytest.raises(ValueError, match='parity failed'):
        bounded_error(left, np.array([1000.002, 0.]), 1e-3)
    with pytest.raises(ValueError, match='parity failed'):
        bounded_error(np.array([0.]), np.array([.00002]), 1e-5)


@pytest.mark.parametrize('fault', ['shape', 'left_nan', 'right_inf'])
def test_invalid_reference_arrays_are_rejected(fault):
    np = pytest.importorskip('numpy')
    left = np.zeros(3); right = left.copy()
    if fault == 'shape': right = right[:, None]
    if fault == 'left_nan': left[0] = np.nan
    if fault == 'right_inf': right[0] = np.inf
    with pytest.raises(ValueError, match='Invalid numerical'):
        bounded_error(left, right, 1e-3)
