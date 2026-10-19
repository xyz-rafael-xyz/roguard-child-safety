import random

import pytest

from training.context_consistency_v9 import plan_batches
from training.context_transition_v9 import generate


def test_minibatches_preserve_every_case_and_history_family():
    rows = generate('uk', 'train')
    batches = plan_batches(rows, random.Random(1162))
    indices = [i for ix, _, _ in batches for i in ix]
    assert sorted(indices) == list(range(2*len(rows)))
    assert all(len(ix) == 12 and len(invariant) == 12 for ix, pairs, invariant in batches)
    assert sum(len(pairs) for ix, pairs, invariant in batches) == 216


def test_consistency_penalty_is_zero_for_equal_scores_and_has_finite_gradients():
    torch = pytest.importorskip('torch')
    from training.context_consistency_v9 import loss
    logits = torch.tensor([[2., 0., -1.], [2., 0., -1.]], requires_grad=True)
    labels = torch.tensor([0, 0])
    weights = torch.ones(2)
    baseline = loss(logits, labels, weights, [], [])
    assert torch.allclose(baseline, loss(logits, labels, weights, [], [(0, 1)]))
    changed = torch.tensor([[2., 0., -1.], [-1., 1., 2.]], requires_grad=True)
    combined = loss(changed, labels, weights, [], [(0, 1)])
    assert combined > loss(changed, labels, weights, [], [])
    combined.backward()
    assert torch.isfinite(changed.grad).all()
