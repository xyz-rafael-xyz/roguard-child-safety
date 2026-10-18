from types import SimpleNamespace
import pytest

from roguard.native_fields import _native_states
from training.native_fields_v9 import SIZES, network


@pytest.mark.parametrize('architecture',['bert','roberta'])
def test_binary_and_semantic_objectives_train_selected_layers_and_both_heads(architecture):
    torch=pytest.importorskip('torch'); transformers=pytest.importorskip('transformers')
    config_type=getattr(transformers,'BertConfig' if architecture=='bert' else 'RobertaConfig')
    model_type=getattr(transformers,'BertForSequenceClassification' if architecture=='bert' else 'RobertaForSequenceClassification')
    model=model_type(config_type(vocab_size=32,hidden_size=16,num_hidden_layers=6,
        num_attention_heads=4,intermediate_size=32,num_labels=2)).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    layers=model.base_model.encoder.layer
    modules=[*layers[-4:],model.classifier]
    if model.base_model.pooler is not None: modules.append(model.base_model.pooler)
    for module in modules:
        for parameter in module.parameters(): parameter.requires_grad_(True)
    hidden,binary=_native_states(SimpleNamespace(model=model),dict(input_ids=torch.tensor([[3,4,5,6]]),
        attention_mask=torch.ones(1,4,dtype=torch.long)),include_logits=True)
    head=network(torch,32,16)
    fields=torch.split(head(torch.cat((hidden[:,0],hidden[:,2:].mean(dim=1)),dim=1)),SIZES,dim=-1)
    expected=torch.zeros(1,dtype=torch.long)
    loss=torch.nn.functional.cross_entropy(binary,expected)+sum(
        torch.nn.functional.cross_entropy(value,expected) for value in fields)/len(SIZES)
    loss.backward()
    assert all(p.grad is None for layer in layers[:2] for p in layer.parameters())
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()) for module in modules)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
