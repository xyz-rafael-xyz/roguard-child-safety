from copy import deepcopy
import pytest


@pytest.mark.parametrize('tied',[True,False])
@pytest.mark.parametrize('cap',[None,3.0])
def test_gemma4_last_token_and_cached_projection_match_complete_architecture(tied,cap):
    mx=pytest.importorskip('mlx.core')
    from mlx_lm.models.gemma4_text import Model,ModelArgs
    from mlx_lm.models.cache import make_prompt_cache
    from training.gemma4_projection_v9 import last_logits
    mx.set_default_device(mx.cpu)
    mx.random.seed(139)
    args=ModelArgs(hidden_size=16,num_hidden_layers=2,intermediate_size=32,
                   num_attention_heads=2,head_dim=8,global_head_dim=8,
                   num_key_value_heads=1,num_kv_shared_layers=0,vocab_size=64,
                   vocab_size_per_layer_input=64,hidden_size_per_layer_input=8,
                   layer_types=['sliding_attention','full_attention'],
                   final_logit_softcapping=cap,tie_word_embeddings=tied)
    model=Model(args)
    tokens=mx.array([[1,2,3,4]])
    full=model(tokens)[:,-1:,:]
    projected=last_logits(model,tokens)
    cache=make_prompt_cache(model)
    mx.eval(model.model(tokens[:,:2],cache=cache))
    cached=last_logits(model,tokens[:,2:],cache=deepcopy(cache))
    mx.eval(full,projected,cached)
    assert float(mx.max(mx.abs(full-projected)).item())<1e-5
    assert float(mx.max(mx.abs(full-cached)).item())<1e-5
    if cap is not None:
        assert float(mx.max(mx.abs(projected)).item())<=cap
