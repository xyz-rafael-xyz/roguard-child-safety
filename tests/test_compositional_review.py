import pytest

from training.prepare_compositional_review_v10 import build,scoped,SCOPES
from training.extract_large_review_v10 import validate_new
from training.prepare_instruction_sft_v8 import target


def test_review_compositions_preserve_family_isolation_and_ontology_labels():
    populations,cores,stats=build();family_partition={}
    assert {key:value['rows'] for key,value in stats.items()}=={'train':4608,'dev':576,'test':1728}
    for split,rows in populations.items():
        for row in rows:
            assert family_partition.setdefault(row['family'],split)==split
            assert row['review']==target(row)
            if row['attributes']['concern']=='1':assert row['review']==int(row['scope'] not in ('negated','fiction'))
            else:assert row['scope']=='affirmed' and row['review']==0
            if row['scope']=='retracted':
                assert row['turns'][-2]['role']=='minor'
                assert row['turns'][-2]['text'] in row['turns'][-1]['text']
                assert row['attributes']['relation']=='RETRACTS'
    assert not cores['train']&cores['dev'] and not cores['train']&cores['test']
    validate_new(populations['train']+populations['dev'])
    with pytest.raises(ValueError,match='Only TRAIN/DEV'):validate_new(populations['test'])


def test_benign_negation_is_not_automatically_labeled_benign():
    for scope in SCOPES[1:]:
        with pytest.raises(ValueError,match='ordinary functioning'):scoped('Un mesaj obișnuit suficient de lung.','ro',scope,False)


def test_large_review_extraction_rejects_duplicate_or_nonfocal_inputs():
    populations,_,_=build();row=populations['train'][0]
    with pytest.raises(ValueError,match='Duplicate new payload'):validate_new([row,row])
    with pytest.raises(ValueError,match='focal-last'):validate_new([{**row,'focus':0}])
