import hashlib
import json

import pytest

from scripts.package_response_head_v10 import export


def source(tmp_path):
    study=tmp_path/'study';study.mkdir();weights=tmp_path/'weights.safetensors';weights.write_bytes(b'synthetic-test-bytes')
    selected={'configuration':{'hidden':0},'head_sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}
    selection={'complete':True,'selected':selected};(study/'selection.json').write_text(json.dumps(selection))
    report={'complete':True,'selected':selected,'selection_sha256':hashlib.sha256((study/'selection.json').read_bytes()).hexdigest()}
    (study/'report.json').write_text(json.dumps(report));return study,weights


def test_export_is_bound_research_artifact_and_never_claims_runtime_verification(tmp_path):
    study,weights=source(tmp_path);output=tmp_path/'export';manifest=export(study,weights,output)
    assert (output/'head.safetensors').read_bytes()==weights.read_bytes()
    assert manifest['research_only'] and manifest['numerical_runtime_verification_required']
    assert not manifest['stable_release_ready']
    assert json.loads((output/'response.json').read_text())['qualified'] is False
    with pytest.raises(ValueError,match='overwrite'):export(study,weights,output)


@pytest.mark.parametrize('mutation',['weights','selection','incomplete'])
def test_changed_or_incomplete_selection_is_not_exported(tmp_path,mutation):
    study,weights=source(tmp_path)
    if mutation=='weights':weights.write_bytes(b'changed')
    elif mutation=='selection':(study/'selection.json').write_text('{}')
    else:
        report=json.loads((study/'report.json').read_text());report['complete']=False
        (study/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):export(study,weights,tmp_path/'export')
    assert not (tmp_path/'export').exists()
