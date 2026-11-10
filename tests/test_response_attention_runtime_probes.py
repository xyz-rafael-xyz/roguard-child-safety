import pytest
from eval.verify_response_attention_runtime_v10 import development_rows
from roguard.assurance import digest


def example(partition='dev'):
    row={'case_id':'case1','partition':partition,'language':'ro','context':'Context','draft':'Răspuns'}
    identity={'case_id':row['case_id'],'partition':partition,'input_sha256':digest({'language':'ro','turns':[
        {'role':'minor','text':'Context'},{'role':'assistant','text':'Răspuns'}]})}
    return row,identity


def test_reference_mapping_rejects_reserved_duplicate_and_changed_inputs():
    row,identity=example();assert development_rows([row],[identity])==[row]
    with pytest.raises(ValueError,match='mapping'):development_rows([{**row,'context':'Changed'}],[identity])
    with pytest.raises(ValueError,match='duplicate'):development_rows([row,row],[identity,identity])
    with pytest.raises(ValueError,match='Incomplete'):development_rows([row],[])
    row,identity=example('test')
    with pytest.raises(ValueError,match='Reserved'):development_rows([row],[identity])
