import json

import pytest
from scripts.package_response_attention_v10 import export
from roguard.gemma4_response import PROMPT,LABELS
from roguard.gemma4_contract import BASE_MODEL,BASE_REVISION
from roguard.trajectory_data import file_sha


def source(tmp_path):
    study=tmp_path/'study';study.mkdir();weights=tmp_path/'weights';weights.write_bytes(b'fake-test-weights')
    selected={'epoch':3,'adapter_sha256':file_sha(weights)}
    (study/'selection.json').write_text(json.dumps({'complete':True,'selected':selected}))
    (study/'protocol.json').write_text(json.dumps({'prompt':PROMPT}))
    model={'format':'roguard-response-attention-research-1','base_model':BASE_MODEL,'base_revision':BASE_REVISION,
           'prompt':PROMPT,'labels':list(LABELS),'boundary':24,'head':{'hidden':0},'qualified':False,
           'adapter_sha256':file_sha(weights),'protocol_sha256':file_sha(study/'protocol.json'),
           'selection_sha256':file_sha(study/'selection.json')}
    (study/'model.json').write_text(json.dumps(model))
    report={'complete':True,'selected':selected,'selection_sha256':file_sha(study/'selection.json'),
            'model_spec_sha256':file_sha(study/'model.json')}
    (study/'report.json').write_text(json.dumps(report));return study,weights


def test_export_retains_bytes_without_qualification_claim(tmp_path):
    study,weights=source(tmp_path);output=tmp_path/'export';result=export(study,weights,output)
    assert result['research_only'] and result['numerical_runtime_verification_required']
    assert not result['stable_release_ready']
    assert (output/'adapter.safetensors').read_bytes()==weights.read_bytes()
    assert json.loads((output/'response.json').read_text())['qualified'] is False
    with pytest.raises(ValueError,match='overwrite'):export(study,weights,output)


@pytest.mark.parametrize('target',['weights','selection.json','protocol.json','model.json','report.json'])
def test_changed_bound_source_cannot_be_exported(tmp_path,target):
    study,weights=source(tmp_path)
    if target=='weights':weights.write_bytes(b'changed')
    elif target=='report.json':
        p=study/target;data=json.loads(p.read_text());data['complete']=False;p.write_text(json.dumps(data))
    else:(study/target).write_text('{}')
    with pytest.raises(ValueError):export(study,weights,tmp_path/'export')
    assert not (tmp_path/'export').exists()
