"""Prospective assertion-only distillation ablation; never admit data here."""

from roguard.native_fields import FIELDS, SIZES


MODES = ('full_teacher', 'mask_wrong_assertion')
ASSERTION_INDEX = FIELDS.index('assertion')
ASSERTION_START = sum(SIZES[:ASSERTION_INDEX])
TERMS = len(SIZES) + 1


def training_teacher_mask(rows, teacher_logits, mode):
    """Mask only the assertion KL term when frozen teacher and TRAIN disagree.

    All other field terms and binary distillation keep their original weight.
    Rows must already belong to the caller's separately admitted fitting set.
    A fitting partition alone is not evidence of semantic admission.
    """
    import numpy as np

    if mode not in MODES:
        raise ValueError('Unknown assertion distillation mode')
    if not isinstance(rows, list) or not rows:
        raise ValueError('No fitting rows')
    if (not isinstance(teacher_logits, np.ndarray)
            or teacher_logits.shape != (len(rows), sum(SIZES) + 2)
            or teacher_logits.dtype.kind != 'f'
            or not np.isfinite(teacher_logits).all()):
        raise ValueError('Invalid frozen teacher logits')
    for row in rows:
        fields = row.get('fields')
        if row.get('partition') != 'train':
            raise ValueError('Teacher masks may use only admitted TRAIN rows')
        if (not isinstance(fields, list) or len(fields) != len(SIZES)
                or any(type(v) is not int or not 0 <= v < size
                       for v, size in zip(fields, SIZES))):
            raise ValueError('Invalid TRAIN fields')
    mask = np.ones(len(rows), dtype=np.bool_)
    if mode == 'mask_wrong_assertion':
        teacher = teacher_logits[:, ASSERTION_START:ASSERTION_START + SIZES[ASSERTION_INDEX]].argmax(axis=1)
        labels = np.array([row['fields'][ASSERTION_INDEX] for row in rows])
        mask[:] = teacher == labels
    return mask


def teacher_kl_per_row(mx, student_logits, teacher_logits, mask):
    """Return seven-term teacher KL with a fixed denominator, including zeros.

    The one-dimensional boolean mask controls ONLY assertion distillation.
    The TRAIN-only mask is computed before fitting, using frozen teacher
    outputs. Never renormalize over the remaining terms: that would increase
    the strength of unrelated field/binary distillation in masked rows.
    """
    if (len(student_logits.shape) != 2 or student_logits.shape[1] != sum(SIZES) + 2
            or teacher_logits.shape != student_logits.shape
            or mask.shape != (student_logits.shape[0],) or mask.dtype != mx.bool_):
        raise ValueError('Distillation shapes or boolean assertion mask differ')
    divergences = []
    offset = 0
    for size in (*SIZES, 2):
        old = teacher_logits[:, offset:offset + size]
        new = student_logits[:, offset:offset + size]
        old_logp = old - mx.logsumexp(old, axis=-1, keepdims=True)
        new_logp = new - mx.logsumexp(new, axis=-1, keepdims=True)
        term = (mx.exp(old_logp) * (old_logp - new_logp)).sum(axis=-1)
        if offset == ASSERTION_START:
            term = term * mask
        divergences.append(term)
        offset += size
    return mx.stack(divergences, axis=1).sum(axis=1) / TERMS
