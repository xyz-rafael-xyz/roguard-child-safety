import pytest


def test_portable_structured_readout_matches_eval_network():
    np=pytest.importorskip('numpy')
    torch=pytest.importorskip('torch')
    from training.gemma4_structured_v9 import network,portable_logits
    from training.gemma4_readout_v9 import normalize
    torch.manual_seed(143)
    model=network(torch,16,hidden=8).float().eval()
    values=np.random.default_rng(143).normal(size=(19,16)).astype(np.float32)
    with torch.inference_mode():
        reference=model(torch.tensor(normalize(values),dtype=torch.float32)).numpy()
    tensors={k:v.detach().numpy() for k,v in model.state_dict().items()}
    actual=portable_logits(values,tensors)
    assert np.max(np.abs(actual-reference))<1e-6
