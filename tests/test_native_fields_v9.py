import pytest
np=pytest.importorskip('numpy')
from training.native_fields_v9 import policy_mass,FIELDS,SIZES,WORLD_INDICES,WORLD_ACTIONS,fit_couplings
from roguard.native_fields import factorized_review_mass


def test_policy_mass_matches_all_432_deterministic_worlds():
    values=[np.eye(size)[[w[i] for w in WORLD_INDICES]] for i,size in enumerate(SIZES)]
    np.testing.assert_allclose(policy_mass(values),WORLD_ACTIONS,atol=1e-12)


def test_uniform_world_distribution_and_invalid_inputs():
    values=[np.full((2,size),1/size) for size in SIZES]
    np.testing.assert_allclose(policy_mass(values),sum(WORLD_ACTIONS)/len(WORLD_ACTIONS),atol=1e-12)
    values[0][0,0]=np.nan
    with pytest.raises(ValueError):policy_mass(values)


def test_analytic_batch_and_runtime_match_exhaustive_random_marginals():
    rng=np.random.default_rng(943)
    values=[rng.dirichlet(np.ones(size),size=50) for size in SIZES]
    reference=np.zeros(50)
    for world,action in zip(WORLD_INDICES,WORLD_ACTIONS):
        mass=np.ones(50)
        for p,index in zip(values,world):mass*=p[:,index]
        reference+=action*mass
    np.testing.assert_allclose(policy_mass(values),reference,atol=2e-15,rtol=0)
    actual=[factorized_review_mass({f:p[i].tolist() for f,p in zip(FIELDS,values)}) for i in range(50)]
    np.testing.assert_allclose(actual,reference,atol=2e-15,rtol=0)


def test_pair_coupled_closed_form_matches_complete_world_distribution():
    rng=np.random.default_rng(946)
    values=[rng.dirichlet(np.ones(size),size=50) for size in SIZES]
    coupling={'scope':np.exp(rng.normal(size=(3,3))).tolist(),
              'assertion_relation':np.exp(rng.normal(size=(4,2))).tolist()}
    reference=np.zeros(50);normalizer=np.zeros(50)
    for world,action in zip(WORLD_INDICES,WORLD_ACTIONS):
        labels=dict(zip(FIELDS,world));mass=np.ones(50)
        for p,index in zip(values,world):mass*=p[:,index]
        mass*=coupling['scope'][labels['subject']][labels['source']]
        mass*=coupling['assertion_relation'][labels['assertion']][labels['relation']]
        reference+=action*mass;normalizer+=mass
    reference/=normalizer
    np.testing.assert_allclose(policy_mass(values,coupling),reference,atol=2e-15,rtol=0)
    actual=[factorized_review_mass({f:p[i].tolist() for f,p in zip(FIELDS,values)},couplings=coupling) for i in range(50)]
    np.testing.assert_allclose(actual,reference,atol=2e-15,rtol=0)


def test_training_coupling_smoothing_preserves_unseen_pairs():
    from roguard.evidence import ONTOLOGY
    rows=[]
    for world in WORLD_INDICES[:20]:
        rows.append({'attributes':{f:ONTOLOGY[f][i] for f,i in zip(FIELDS,world)}})
    tables=fit_couplings(rows,[1.]*len(rows))
    assert all(np.isfinite(matrix).all() and (np.asarray(matrix)>0).all() for matrix in tables.values())


@pytest.mark.parametrize('architecture',['bert','roberta'])
def test_final_layer_extraction_matches_full_tiny_architecture(architecture):
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    from types import SimpleNamespace
    from roguard.native_fields import _native_states
    config_class=transformers.BertConfig if architecture=='bert' else transformers.RobertaConfig
    model_class=transformers.BertForSequenceClassification if architecture=='bert' else transformers.RobertaForSequenceClassification
    torch.manual_seed(943)
    config=config_class(vocab_size=32,hidden_size=12,num_hidden_layers=2,num_attention_heads=3,
        intermediate_size=24,max_position_embeddings=32,num_labels=2)
    model=model_class(config).float().eval()
    inputs={'input_ids':torch.arange(2,10).repeat(2,1),'attention_mask':torch.ones((2,8),dtype=torch.long)}
    with torch.inference_mode():
        reference=model(**inputs,output_hidden_states=True)
        hidden,logits=_native_states(SimpleNamespace(model=model),inputs,include_logits=True)
    torch.testing.assert_close(hidden,reference.hidden_states[-1],rtol=0,atol=0)
    torch.testing.assert_close(logits,reference.logits,rtol=0,atol=0)
