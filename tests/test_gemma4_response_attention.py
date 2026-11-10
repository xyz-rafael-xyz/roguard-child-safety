"""Public attention-adapter contract, role separation and exception restoration."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from roguard.assurance import policy_digest
from roguard.gemma4_contract import BASE_MODEL,BASE_REVISION,POLICY_SHA256
from roguard.gemma4_review import Gemma4ReviewModel,FORMAT as REVIEW_FORMAT,LORA
import roguard.gemma4_response_attention as module


def configured(tmp_path,change=None):
    reference_spec={'format':REVIEW_FORMAT,'language':'ro','base_model':BASE_MODEL,'base_revision':BASE_REVISION,
        'policy_sha256':POLICY_SHA256,'policy_digest':policy_digest(),'adapter_sha256':'1'*64,
        'mode':'binary','couplings':None,'threshold':.5,'boundary':24,'lora':LORA}
    reference_path=tmp_path/'reference.json';reference_path.write_text(json.dumps(reference_spec))
    reference=Gemma4ReviewModel(reference_path,tmp_path/'binary.safetensors',tmp_path/'base',
        expected_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
    spec={'format':module.FORMAT,'base_model':BASE_MODEL,'base_revision':BASE_REVISION,
        'query_sha256':module.QUERY_SHA256,'adapter_sha256':'2'*64,'boundary':24,'hidden':0,
        'labels':list(module.LABELS),'qualified':False}
    if change:spec.update(change)
    path=tmp_path/'response.json';path.write_text(json.dumps(spec))
    return module.Gemma4ResponseAttentionModel(path,tmp_path/'response.safetensors',reference,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def backend(monkeypatch,features='features',failure=None):
    calls=[]
    class Engine:
        def __init__(self,*args):
            calls.append('engine');self.model=SimpleNamespace(base=object())
            self.mx=SimpleNamespace(clear_cache=lambda:calls.append('clear'))
    class Readout:
        def __init__(self,*args):self.adapters=object();calls.append('readout')
        def logits(self,x):return [1.,0.,0.,0.,0.]
    @contextmanager
    def replace(*args):
        calls.append('activate')
        try:yield
        finally:calls.append('restore')
    def extract(engine,payload):
        calls.append(deepcopy(payload))
        if failure:raise failure
        return features
    monkeypatch.setattr(module,'_Engine',Engine);monkeypatch.setattr(module,'_AdapterReadout',Readout)
    monkeypatch.setattr(module,'_replace_adapters',replace);monkeypatch.setattr(module,'_features',extract)
    return calls


def test_contract_roles_and_encoder_reuse(tmp_path,monkeypatch):
    calls=backend(monkeypatch);model=configured(tmp_path)
    output=model.review_response('O replică.','ro','Un context.')
    assert output['label']=='appropriate' and not output['review_requested']
    assert calls[3]['turns']==[{'role':'minor','text':'Un context.'},{'role':'assistant','text':'O replică.'}]
    assert calls[-2:]==['restore','clear']
    assert all(output[key] is False for key in ('qualified','safe_response_established','permits_external_action',
        'permits_notification','probabilities_calibrated','text_in_report','contextual_factual_support_established'))
    assert 'O replică' not in json.dumps(output,ensure_ascii=False)
    model.review_response('Інша репліка.','uk')
    assert calls.count('engine')==1 and calls.count('readout')==1
    model.reference._engine=None;model.review_response('Alt text.','ro')
    assert calls.count('engine')==2 and calls.count('readout')==1


def test_failure_restores_then_clears(tmp_path,monkeypatch):
    calls=backend(monkeypatch,failure=RuntimeError('deliberate'))
    with pytest.raises(RuntimeError,match='deliberate'):configured(tmp_path).review_response('Text.','ro')
    assert calls[-2:]==['restore','clear']


def test_token_overflow_abstains_and_restores(tmp_path,monkeypatch):
    calls=backend(monkeypatch,features=None);output=configured(tmp_path).review_response('Text.','uk')
    assert output['abstained'] and output['review_requested'] and output['probabilities'] is None
    assert output['reason']=='INPUT_EXCEEDS_MODEL_CONTEXT' and calls[-2:]==['restore','clear']


@pytest.mark.parametrize('change',[{'qualified':True},{'hidden':True},{'hidden':128},{'boundary':True},
    {'boundary':23},{'adapter_sha256':'Z'*64},{'labels':list(reversed(module.LABELS))},
    {'query_sha256':'0'*64},{'base_revision':'wrong'},{'extra':'field'}])
def test_incompatible_artifact_rejected(tmp_path,change):
    with pytest.raises(ValueError):configured(tmp_path,change)


def test_reference_change_rejected_before_loading(tmp_path,monkeypatch):
    calls=backend(monkeypatch);model=configured(tmp_path);model.reference._spec['threshold']=.7
    with pytest.raises(ValueError,match='Reference configuration'):model.review_response('Text.','ro')
    assert calls==[]


@pytest.mark.parametrize('draft,language,context',[('','ro',''),('x','en',''),('x','ro',None),('x'*6001,'ro',''),('x','ro','a'*12001),('\ud800','ro','')])
def test_invalid_request_rejected_before_loading(tmp_path,monkeypatch,draft,language,context):
    calls=backend(monkeypatch)
    with pytest.raises(ValueError):configured(tmp_path).review_response(draft,language,context)
    assert calls==[]
