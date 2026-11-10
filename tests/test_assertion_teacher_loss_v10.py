import copy

import pytest

from training.assertion_teacher_loss_v10 import (
    ASSERTION_INDEX, ASSERTION_START, TERMS, teacher_kl_per_row, training_teacher_mask,
)
from roguard.native_fields import SIZES


def fixture():
    np = pytest.importorskip('numpy')
    rows = [{'partition': 'train', 'fields': [0, 0, assertion, 0, 0, 0]}
            for assertion in range(4)]
    teacher = np.zeros((4, sum(SIZES) + 2), dtype=np.float32)
    for row, predicted in enumerate((0, 2, 0, 3)):
        teacher[row, ASSERTION_START + predicted] = 2.
    return np, rows, teacher


def test_mask_only_assertion_and_keeps_input_unchanged():
    np, rows, teacher = fixture()
    before_rows, before_teacher = copy.deepcopy(rows), teacher.copy()
    full = training_teacher_mask(rows, teacher, 'full_teacher')
    masked = training_teacher_mask(rows, teacher, 'mask_wrong_assertion')
    expected = np.array([True, False, False, True], dtype=np.bool_)
    np.testing.assert_array_equal(masked, expected)
    np.testing.assert_array_equal(full, np.ones_like(expected))
    assert rows == before_rows
    np.testing.assert_array_equal(teacher, before_teacher)


@pytest.mark.parametrize('partition', ['dev', 'test', 'fit', 'probe_operator', 'probe_meaning', 'probe_combination', 'probe_both'])
def test_reject_non_train_labels_before_masking(partition):
    _, rows, teacher = fixture()
    rows[2]['partition'] = partition
    with pytest.raises(ValueError, match='only admitted TRAIN'):
        training_teacher_mask(rows, teacher, 'mask_wrong_assertion')


@pytest.mark.parametrize('corruption', ['nan', 'infinity', 'integer', 'row_count', 'logit_count', 'boolean_field', 'field_range', 'missing_field', 'mode', 'empty'])
def test_reject_invalid_inputs(corruption):
    np, rows, teacher = fixture()
    mode = 'mask_wrong_assertion'
    if corruption == 'nan': teacher[0, 0] = np.nan
    elif corruption == 'infinity': teacher[0, 0] = np.inf
    elif corruption == 'integer': teacher = teacher.astype(np.int32)
    elif corruption == 'row_count': teacher = teacher[:-1]
    elif corruption == 'logit_count': teacher = teacher[:, :-1]
    elif corruption == 'boolean_field': rows[0]['fields'][0] = False
    elif corruption == 'field_range': rows[0]['fields'][2] = 4
    elif corruption == 'missing_field': rows[0]['fields'] = rows[0]['fields'][:-1]
    elif corruption == 'mode': mode = 'mask_all_fields'
    elif corruption == 'empty': rows = []
    with pytest.raises(ValueError):
        training_teacher_mask(rows, teacher, mode)


def test_kl_value_and_gradient_match_independent_reference():
    np, rows, teacher = fixture()
    mx = pytest.importorskip('mlx.core')
    mx.set_default_device(mx.cpu)
    student = np.random.default_rng(111).normal(size=teacher.shape).astype(np.float32)
    full = training_teacher_mask(rows, teacher, 'full_teacher')
    mask = training_teacher_mask(rows, teacher, 'mask_wrong_assertion')
    boundaries = np.cumsum((0, *SIZES, 2))
    reference = np.zeros(len(rows), dtype=np.float64)
    expected_gradient = np.zeros_like(student, dtype=np.float64)
    reference_full = reference.copy()
    for j, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        def probabilities(x):
            v = x.astype(np.float64)
            e = np.exp(v - v.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        p = probabilities(teacher[:, start:end])
        q = probabilities(student[:, start:end])
        term = (p * (np.log(p) - np.log(q))).sum(axis=1) / 7
        reference_full += term
        weight = mask if j == ASSERTION_INDEX else np.ones(len(rows))
        reference += term * weight
        expected_gradient[:, start:end] = (q-p) * weight[:, None] / 7
    old, weights = mx.array(teacher), mx.array(mask)
    actual = teacher_kl_per_row(mx, mx.array(student), old, weights)
    gradient = mx.grad(lambda z: teacher_kl_per_row(mx, z, old, weights).sum())(mx.array(student))
    actual_full = teacher_kl_per_row(mx, mx.array(student), old, mx.array(full))
    np.testing.assert_allclose(np.asarray(actual), reference, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(actual_full), reference_full, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(gradient), expected_gradient, rtol=1e-5, atol=1e-7)
    assert np.count_nonzero(np.asarray(gradient)[[1, 2], ASSERTION_START:ASSERTION_START+4]) == 0
    # All six other terms retain 1/7 strength, including the binary term.
    assert np.any(np.asarray(gradient)[:, -2:] != 0)
    assert TERMS == 7


@pytest.mark.parametrize('target', ['student', 'teacher', 'mask'])
def test_distillation_rejects_bad_shapes(target):
    _, rows, teacher = fixture()
    mx = pytest.importorskip('mlx.core')
    mx.set_default_device(mx.cpu)
    values = dict(student=mx.array(teacher), teacher=mx.array(teacher),
                  mask=mx.array(training_teacher_mask(rows, teacher, 'full_teacher')))
    values[target] = values[target][:-1] if target == 'mask' else values[target][:, :-1]
    with pytest.raises(ValueError, match='shapes'):
        teacher_kl_per_row(mx, values['student'], values['teacher'], values['mask'])


@pytest.mark.parametrize('bad', ['full_matrix', 'zero', 'negative', 'fraction', 'nan'])
def test_loss_cannot_mask_other_fields_or_accept_numeric_weights(bad):
    np, rows, teacher = fixture()
    mx = pytest.importorskip('mlx.core')
    mx.set_default_device(mx.cpu)
    mask = np.ones((4, 7) if bad == 'full_matrix' else (4,), dtype=np.float32)
    if bad == 'zero': mask[0] = 0.
    elif bad == 'negative': mask[0] = -1.
    elif bad == 'fraction': mask[0] = .5
    elif bad == 'nan': mask[0] = np.nan
    with pytest.raises(ValueError, match='boolean assertion mask'):
        teacher_kl_per_row(mx, mx.array(teacher), mx.array(teacher), mx.array(mask))
