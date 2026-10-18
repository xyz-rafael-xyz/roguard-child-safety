from types import SimpleNamespace
import pytest

from roguard.evidence import ONTOLOGY
from roguard.native_fields import _native_states
from roguard.native_review import prepare_input
from training.joint_semantic_data_v9 import generate
from training.native_fields_v9 import SIZES, network
from training.threshold_margin_fields_v9 import margin_threshold


def test_numeric_gap_retains_required_positives_and_resists_roundoff():
    scores=[0.,.1,.49995,.5,.50001,.8]
    cutoff=margin_threshold(scores,.49999)
    assert all(v>=cutoff for v in scores if v>=.49999)
    assert [v>=cutoff for v in scores]==[min(1.,max(0.,v-9e-5))>=cutoff for v in scores]
    assert cutoff==pytest.approx((.1+.49995)/2)


def test_no_safe_gap_uses_explicit_all_review_boundary():
    assert margin_threshold([0.,1e-6],0.)==0.


@pytest.mark.parametrize('language',['ro','uk'])
def test_authored_training_compositions_have_consistent_semantics(language):
    rows=generate(language);seen={}
    assert len(rows)==5184
    for row in rows:
        assert set(row['attributes'])==set(ONTOLOGY)
        assert all(row['attributes'][f] in ONTOLOGY[f] for f in ONTOLOGY)
        assert (row['attributes']['assertion']=='RETRACTED') == (row['attributes']['relation']=='RETRACTS')
        key=prepare_input(dict(language=language,turns=row['turns']),row['focus'])[0]
        assert key not in seen or seen[key]==row['attributes']
        seen[key]=row['attributes']
    assert len(seen)==len(rows)


@pytest.mark.parametrize('architecture',['bert','roberta'])
def test_joint_loss_reaches_only_selected_encoder_layers(architecture):
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    config_type=getattr(transformers,'BertConfig' if architecture=='bert' else 'RobertaConfig')
    model_type=getattr(transformers,'BertForSequenceClassification' if architecture=='bert' else 'RobertaForSequenceClassification')
    config=config_type(vocab_size=32,hidden_size=16,num_hidden_layers=4,num_attention_heads=4,
                       intermediate_size=32,num_labels=2)
    model=model_type(config).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    layers=model.base_model.encoder.layer
    for layer in layers[-2:]:
        for parameter in layer.parameters():parameter.requires_grad_(True)
    engine=SimpleNamespace(model=model)
    hidden,_=_native_states(engine,dict(input_ids=torch.tensor([[3,4,5,6]]),attention_mask=torch.ones(1,4,dtype=torch.long)))
    features=torch.cat((hidden[:,0,:],hidden[:,2:,:].mean(dim=1)),dim=1)
    head=network(torch,32,16)
    logits=torch.split(head(features),SIZES,dim=-1)
    loss=sum(torch.nn.functional.cross_entropy(value,torch.zeros(1,dtype=torch.long)) for value in logits)/len(SIZES)
    loss.backward()
    assert all(p.grad is None for layer in layers[:2] for p in layer.parameters())
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in layer.parameters()) for layer in layers[-2:])
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
