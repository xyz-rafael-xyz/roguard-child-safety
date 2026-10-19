import pytest
from training.scope_distillation_data_v9 import generate, anchored_loss
from training.train_bare_distress_v9 import texts, read
from training.prepare_instruction_sft_v8 import target
from pathlib import Path


def test_authored_data_covers_scope_without_protected_input_overlap():
    rows = generate()
    assert len(rows) == 7200
    assert len({texts(r) for r in rows}) == len(rows)
    assert len({r['content_family'] for r in rows}) == 96
    assert {r['attributes']['assertion'] for r in rows} == {'AFFIRMED', 'NEGATED', 'UNCERTAIN', 'RETRACTED'}
    for row in rows:
        attrs = row['attributes']
        expected = attrs['concern'] == '1' and (attrs['assertion'] == 'RETRACTED'
                   or (attrs['source'] != 'FICTION' and attrs['assertion'] != 'NEGATED'))
        assert target(row) == int(expected)
    root = Path('research/v9-stabilization/bare-distress-95/uk')
    protected = {texts(r) for split in ('dev', 'cal', 'stress') for r in read(root/(split+'.jsonl.gz'))}
    assert not protected.intersection(texts(r) for r in rows)


def test_distillation_never_changes_unanchored_gradients_or_trains_teacher():
    torch = pytest.importorskip('torch')
    logits = torch.tensor([[.2, -.1], [.1, .4]], requires_grad=True)
    labels = torch.tensor([1, 0])
    teacher = torch.tensor([[.8, .2], [.2, .8]], requires_grad=True)
    primary = torch.nn.functional.cross_entropy(logits, labels)
    reference = torch.autograd.grad(primary, logits, retain_graph=True)[0]
    loss = anchored_loss(logits, labels, teacher, torch.tensor([True, False]))
    loss.backward()
    assert teacher.grad is None
    torch.testing.assert_close(logits.grad[1], reference[1], atol=0, rtol=0)
    assert not torch.allclose(logits.grad[0], reference[0])


def test_no_anchor_rows_reduces_exactly_to_labeled_loss():
    torch = pytest.importorskip('torch')
    logits = torch.tensor([[.2, -.1], [.1, .4]], requires_grad=True)
    labels = torch.tensor([1, 0])
    teacher = torch.tensor([[.8, .2], [.2, .8]])
    result = anchored_loss(logits, labels, teacher, torch.tensor([False, False]))
    torch.testing.assert_close(result, torch.nn.functional.cross_entropy(logits, labels), atol=0, rtol=0)
