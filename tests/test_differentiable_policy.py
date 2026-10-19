import pytest


@pytest.mark.parametrize('coupled', [False, True])
def test_differentiable_policy_matches_independent_numeric_scorer(coupled):
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    from roguard.evidence import ONTOLOGY
    from training.native_fields_v9 import FIELDS, SIZES, policy_mass as reference
    from training.differentiable_policy_v9 import policy_mass
    torch.manual_seed(151)
    logits = torch.randn(8, sum(SIZES), dtype=torch.float64, requires_grad=True)
    couplings = None
    if coupled:
        rng = np.random.default_rng(151)
        couplings = {name: rng.uniform(.1, 3., (len(ONTOLOGY[left]), len(ONTOLOGY[right]))).tolist()
                     for name, left, right in (('scope', 'subject', 'source'), ('assertion_relation', 'assertion', 'relation'))}
    actual = policy_mass(torch, logits, couplings)
    probabilities = [torch.softmax(p, dim=-1).detach().numpy() for p in torch.split(logits, SIZES, dim=-1)]
    assert np.max(np.abs(actual.detach().numpy()-reference(probabilities, couplings))) < 1e-12
    labels = torch.tensor([0., 1.]*4, dtype=torch.float64)
    loss = torch.nn.functional.binary_cross_entropy(actual, labels)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    for field, gradient in zip(FIELDS, torch.split(logits.grad, SIZES, dim=-1)):
        if field == 'time':
            assert torch.count_nonzero(gradient) == 0
        else:
            assert float(gradient.abs().sum()) > 0
    improved = policy_mass(torch, logits.detach()-.1*logits.grad, couplings)
    assert float(torch.nn.functional.binary_cross_entropy(improved, labels)) < float(loss.detach())
