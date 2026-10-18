from copy import deepcopy
from types import SimpleNamespace
import pytest

from roguard.evidence import ONTOLOGY
from training.evaluate_interpreter_batch_v9 import grouped_indices, repeated_prefix
from training.batch_interpreter_gate_v9 import compare


def test_groups_never_mix_token_layouts_or_drop_cases():
    prepared=[([],p) for p in [(2,5),(3,6),(2,5),(2,5),(3,6)]]
    batches=grouped_indices(prepared,2)
    assert sorted(i for batch in batches for i in batch)==list(range(5))
    assert all(len(batch)<=2 and len({tuple(prepared[i][1]) for i in batch})==1 for batch in batches)


def test_prefix_repeat_preserves_offsets_and_does_not_mutate_original():
    np=pytest.importorskip('numpy')
    original=SimpleNamespace(keys=np.ones((1,2,3,4)),values=np.ones((1,2,3,4))*2,offset=3,_idx=3,max_size=1024)
    repeated=repeated_prefix([original],4,np.repeat)[0]
    assert repeated.keys.shape==(4,2,3,4) and (repeated.offset,repeated._idx,repeated.max_size)==(3,3,1024)
    repeated.keys[0,0,0,0]=9
    assert original.keys[0,0,0,0]==1


def row():
    return dict(case_id='one',language='ro',gold=0,predicted=0,valid_label_tokens=True,
        interpretation={field:labels[0] for field,labels in ONTOLOGY.items()},
        conditional_field_probabilities={field:[1.]+[0.]*(len(labels)-1) for field,labels in ONTOLOGY.items()})


def test_identical_complete_decoding_passes():
    assert compare([row()],[row()])['parity_pass']


@pytest.mark.parametrize('change',['decision','field','validity','probability','abstention'])
def test_decoder_changes_cannot_enable_batched_backend(change):
    old=row();new=deepcopy(old)
    if change=='decision':new['predicted']=1
    elif change=='field':new['interpretation']['assertion']='NEGATED'
    elif change=='validity':new['valid_label_tokens']=False
    elif change=='abstention':old['valid_label_tokens']=new['valid_label_tokens']=False
    else:new['conditional_field_probabilities']['concern']=[.99,.01]
    assert not compare([old],[new])['parity_pass']


def test_nonfinite_probabilities_rejected():
    old=row();new=row();new['conditional_field_probabilities']['concern']=[float('nan'),0.]
    with pytest.raises(ValueError):compare([old],[new])


def test_uncached_probe_keeps_full_context_but_projects_one_position():
    np=pytest.importorskip('numpy')
    from training.gemma_projection_v9 import last_logits
    seen=[]
    weights=np.arange(35).reshape(5,7)
    class Transformer:
        def __init__(self):
            self.embed_tokens=SimpleNamespace(as_linear=self.project)
        def __call__(self,tokens,cache=None):
            assert cache is None
            seen.append(tokens.shape)
            return np.repeat(np.cumsum(tokens,axis=1)[...,None],5,axis=-1)
        def project(self,hidden):
            assert hidden.shape[1]==1
            return hidden@weights
    model=SimpleNamespace(model_type='gemma3_text',model=Transformer(),tie_word_embeddings=True)
    tokens=np.arange(24).reshape(2,12)
    actual=last_logits(model,tokens)
    reference=np.repeat(np.cumsum(tokens,axis=1)[...,None],5,axis=-1)@weights
    assert seen==[(2,12)]
    np.testing.assert_array_equal(actual,reference[:,-1:,:])


def test_installed_mlx_cache_repetition_keeps_rows_separate():
    mx=pytest.importorskip('mlx.core')
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    previous=mx.default_device();mx.set_default_device(mx.cpu)
    try:
        for cache in (KVCache(),RotatingKVCache(max_size=16)):
            cache.update_and_fetch(mx.ones((1,2,5,4)),mx.ones((1,2,5,4))*2)
            expanded=repeated_prefix([cache],3,mx.repeat)[0]
            additions=mx.broadcast_to(mx.arange(3).reshape(3,1,1,1),(3,2,3,4)).astype(mx.float32)
            keys,values=expanded.update_and_fetch(additions,additions+10)
            mx.eval(keys,values)
            assert expanded.offset==8 and cache.offset==5
            assert keys.shape==(3,2,8,4)
            assert keys[:,0,5,0].tolist()==[0.,1.,2.]
            assert values[:,0,5,0].tolist()==[10.,11.,12.]
            assert keys[:,0,0,0].tolist()==[1.,1.,1.]
            assert cache.state[0].shape==(1,2,5,4)
    finally: mx.set_default_device(previous)
