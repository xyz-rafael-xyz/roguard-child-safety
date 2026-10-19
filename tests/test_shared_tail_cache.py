import pytest


def test_identity_bound_cache_roundtrip_rejects_mismatch_and_nonfinite_state(tmp_path):
    np = pytest.importorskip('numpy')
    from training.shared_tail_cache_v9 import read_state, write_state
    identity = dict(token_sha256='a'*64, source_freeze_sha256='b'*64)
    state = dict(start=2, layers=4, length=3, previous_kvs=(0, 1),
        hidden=np.ones((1, 1, 8), dtype=np.float32),
        shared={i: (np.ones((1, 1, 3, 4), dtype=np.float32), np.zeros((1, 1, 3, 4), dtype=np.float32)) for i in (0, 1)},
        per_layer=[np.ones((1, 1, 4), dtype=np.float32), None],
        masks=[np.array([[False, True, True]]), None])
    path = tmp_path/'state.npz'
    write_state(path, state, identity)
    actual = read_state(path, expected_identity=identity)
    assert np.array_equal(actual['hidden'], state['hidden'])
    assert np.array_equal(actual['masks'][0], state['masks'][0])
    for i in state['shared']:
        assert all(np.array_equal(a, b) for a, b in zip(state['shared'][i], actual['shared'][i]))
    with pytest.raises(ValueError, match='identity'):
        read_state(path, expected_identity=dict(identity, token_sha256='c'*64))
    state['hidden'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        write_state(tmp_path/'bad.npz', state, identity)


def test_cached_batch_matches_individual_full_sequence_and_rejects_mixed_lengths(tmp_path):
    mx = pytest.importorskip('mlx.core')
    from mlx_lm.models.gemma4_text import Model, ModelArgs
    from training.gemma4_shared_tail_v9 import capture, replay
    from training.shared_tail_cache_v9 import batch_states, read_state, to_mlx, write_state
    mx.set_default_device(mx.cpu)
    model = Model(ModelArgs(hidden_size=16, num_hidden_layers=4, intermediate_size=32,
        num_attention_heads=2, head_dim=8, global_head_dim=8, num_key_value_heads=1,
        num_kv_shared_layers=2, vocab_size=64, vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention']*2))
    tokens = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
    states = []
    for i, row in enumerate(tokens):
        state = capture(model, mx.array([row]), start=2)
        identity = dict(token_sha256=str(i)*64, source_freeze_sha256='b'*64)
        path = tmp_path/f'{i}.npz'
        write_state(path, state, identity)
        states.append(to_mlx(read_state(path, expected_identity=identity)))
    actual = replay(model, batch_states(states))
    expected = model.model(mx.array(tokens))[:, -1:, :]
    mx.eval(actual, expected)
    assert float(mx.max(mx.abs(actual-expected)).item()) < 2e-5
    shorter = capture(model, mx.array([[1, 2]]), start=2)
    with pytest.raises(ValueError, match='positions'):
        batch_states([states[0], shorter])
