"""Portable response contract checks; numerical/backend parity is separate."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from roguard.assurance import policy_digest
from roguard.gemma4_contract import BASE_MODEL,BASE_REVISION,POLICY_SHA256
from roguard.gemma4_review import Gemma4ReviewModel,FORMAT as REVIEW_FORMAT,LORA
from roguard.gemma4_response import Gemma4ResponseModel,FORMAT,LABELS,QUERY_SHA256,PROMPT,_probabilities
import roguard.gemma4_response as module


def configured(tmp_path,modify=None):
    spec={'format':REVIEW_FORMAT,'language':'ro','base_model':BASE_MODEL,'base_revision':BASE_REVISION,
          'policy_sha256':POLICY_SHA256,'policy_digest':policy_digest(),'adapter_sha256':'1'*64,
          'mode':'binary','couplings':None,'threshold':.5,'boundary':24,'lora':LORA}
    path=tmp_path/'reference.json';path.write_text(json.dumps(spec))
    reference=Gemma4ReviewModel(path,tmp_path/'adapter.safetensors',tmp_path/'base',expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    spec={'format':FORMAT,'base_model':BASE_MODEL,'base_revision':BASE_REVISION,'query_sha256':QUERY_SHA256,
          'head_sha256':'2'*64,'hidden':0,'labels':list(LABELS),'qualified':False}
    if modify:modify(spec)
    path=tmp_path/'response.json';path.write_text(json.dumps(spec))
    return Gemma4ResponseModel(path,tmp_path/'head.safetensors',reference,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def backend(monkeypatch,features='features',logits=None):
    calls=[]
    class Engine:
        def __init__(self,*args):
            calls.append('engine');self.mx=SimpleNamespace(clear_cache=lambda:calls.append('clear'))
    class Readout:
        def __init__(self,*args):calls.append('readout')
        def logits(self,value):
            assert value==features
            return logits if logits is not None else [0.,1.,0.,0.,0.]
    def extract(engine,payload):calls.append(deepcopy(payload));return features
    monkeypatch.setattr(module,'_Engine',Engine);monkeypatch.setattr(module,'_Readout',Readout);monkeypatch.setattr(module,'_features',extract)
    return calls


def test_runtime_uses_exact_registered_prompt_and_label_order():
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'training/train_neural_response_v10.py').read_text(encoding='utf-8'))
    registered=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id=='PROMPT' for target in node.targets))
    assert PROMPT==registered
    from roguard.trajectory_model import RESPONSE_LABELS
    assert LABELS==RESPONSE_LABELS


def test_shared_encoder_context_binding_and_no_authority(tmp_path,monkeypatch):
    calls=backend(monkeypatch);model=configured(tmp_path)
    result=model.review_response('O replică de verificat.','ro','Un mesaj anterior.')
    assert result['label']=='boundary' and result['review_requested'] and not result['abstained']
    assert sum(result['probabilities'].values())==pytest.approx(1.)
    assert calls[2]=={'language':'ro','turns':[{'role':'minor','text':'Un mesaj anterior.'},{'role':'assistant','text':'O replică de verificat.'}]}
    assert all(result[key] is False for key in ('qualified','safe_response_established','contextual_factual_support_established','permits_external_action','permits_notification','probabilities_calibrated','text_in_report'))
    assert 'O replică' not in json.dumps(result,ensure_ascii=False)
    model.review_response('Інша репліка для перевірки.','uk')
    assert calls.count('engine')==1 and calls.count('readout')==1 and calls.count('clear')==2
    model.reference._engine=None
    model.review_response('Altă replică.','ro')
    assert calls.count('engine')==2 and calls.count('readout')==1
    assert not hasattr(model._readout,'engine')


def test_context_limit_abstains_and_clears_temporary_cache(tmp_path,monkeypatch):
    calls=backend(monkeypatch,features=None);model=configured(tmp_path)
    result=model.review_response('Un text prea lung pentru model.','ro')
    assert result['abstained'] and result['label'] is None and result['probabilities'] is None
    assert result['review_requested'] and result['reason']=='INPUT_EXCEEDS_MODEL_CONTEXT'
    assert calls[-1]=='clear'


def test_invalid_output_clears_cache_and_raises(tmp_path,monkeypatch):
    calls=backend(monkeypatch,logits=[float('nan')]*5);model=configured(tmp_path)
    with pytest.raises(ValueError,match='logits'):model.review_response('Un text.','ro')
    assert calls[-1]=='clear'


@pytest.mark.parametrize('draft,language,context',[('','ro',''),('x','en',''),('x','ro',None),('x'*6001,'ro',''),('x','ro','a'*12001),('\ud800','ro','')])
def test_invalid_requests_fail_before_model_loading(tmp_path,monkeypatch,draft,language,context):
    calls=backend(monkeypatch);model=configured(tmp_path)
    with pytest.raises(ValueError):model.review_response(draft,language,context)
    assert calls==[]


@pytest.mark.parametrize('key,value',[('qualified',True),('hidden',True),('hidden',256),('labels',list(reversed(LABELS))),('query_sha256','0'*64),('base_revision','wrong'),('head_sha256','Z'*64)])
def test_incompatible_artifacts_are_rejected(tmp_path,key,value):
    with pytest.raises(ValueError):configured(tmp_path,lambda spec:spec.update({key:value}))


def test_reference_mutation_cannot_keep_old_identity(tmp_path,monkeypatch):
    calls=backend(monkeypatch);model=configured(tmp_path);model.reference._spec['threshold']=.7
    with pytest.raises(ValueError,match='Reference configuration'):model.review_response('Un mesaj.','ro')
    assert calls==[]


def test_probability_conversion_is_finite_and_rejects_boolean_or_missing_logits():
    assert _probabilities([1000.,0.,-1000.,0.,0.])[0]==1.
    for logits in ([True,0.,0.,0.,0.],[0.]*4,[float('inf')]*5):
        with pytest.raises(ValueError):_probabilities(logits)
