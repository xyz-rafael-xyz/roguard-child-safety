import pytest

torch = pytest.importorskip('torch')
from training.native_full_optimizer_v9 import parameter_groups, schedule


class Model(torch.nn.Module):
    def __init__(self, pooler):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.embeddings = torch.nn.Linear(3, 3)
        self.base_model.encoder = torch.nn.Module()
        self.base_model.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(3)])
        self.base_model.pooler = torch.nn.Linear(3, 3) if pooler else None
        self.classifier = torch.nn.Linear(3, 2)

    def forward(self, value):
        value = self.base_model.embeddings(value)
        for layer in self.base_model.encoder.layer:
            value = torch.tanh(layer(value))
        if self.base_model.pooler is not None:
            value = torch.tanh(self.base_model.pooler(value))
        return self.classifier(value)


@pytest.mark.parametrize('pooler',[False, True])
def test_every_layer_is_optimized_once_and_schedule_preserves_decay(pooler):
    torch.manual_seed(1133)
    model = Model(pooler)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = parameter_groups(model)
    identities = [id(p) for g in groups for p in g['params']]
    assert len(identities) == len(set(identities)) == len(list(model.parameters()))
    assert all(p.requires_grad for p in model.parameters())
    rates = [g['lr'] for g in groups]
    assert rates == sorted(rates) and rates[0] < rates[-1]
    optimizer = torch.optim.AdamW(groups)
    schedule(optimizer, .3)
    assert [g['lr'] for g in optimizer.param_groups] == [v*.3 for v in rates]
    before = [p.detach().clone() for p in model.parameters()]
    loss = torch.nn.functional.cross_entropy(model(torch.ones(2,3)), torch.tensor([0,1]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(not torch.equal(old, new) for old,new in zip(before,model.parameters()))


def test_omitted_parameters_are_rejected():
    model = Model(False)
    model.extra = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match='every parameter'):
        parameter_groups(model)


def test_tied_parameters_cannot_be_updated_twice():
    model = Model(False)
    model.base_model.encoder.layer[1] = model.base_model.encoder.layer[0]
    with pytest.raises(ValueError, match='multiple optimization groups'):
        parameter_groups(model)
