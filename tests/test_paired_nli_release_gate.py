from copy import deepcopy
import pytest
from training.verify_paired_nli_v9 import qualified as paired_gate
from training.verify_policy_entailment_v9 import qualified as policy_gate
from training.verify_reason_entailment_v9 import qualified as reason_gate


def passing():
    g=dict(positive=100,negative=100,missed=1,false_reviews=5,abstentions=0)
    groups={s:dict(formal=g,conversational=g) if s in ('ro_dev','uk_dev') else dict(all=g) for s in ('ro_dev','uk_dev','ro_scope_dev','uk_scope_dev','uk_stress')}
    audits={s:dict(formal=g,conversational=g) if s in ('ro_cal','uk_cal') else dict(all=g) for s in ('ro_cal','uk_cal','ro_exposed','uk_exposed')}
    return dict(complete=True,counts_pass=True,fresh_test_not_inferred=True,selected_epoch=1,epochs=[dict(epoch=1,metrics=groups)],audit=audits)


@pytest.mark.parametrize("qualified",[paired_gate,policy_gate,reason_gate])
def test_all_strata_and_counts_required(qualified):
    assert qualified(passing())
    for key,value in [('missed',2),('false_reviews',6),('abstentions',1),('positive',0),('negative',0),('missed',False)]:
        record=deepcopy(passing());record['audit']['uk_cal']['conversational'][key]=value
        assert not qualified(record)
    for split in ('ro_scope_dev','uk_stress'):
        record=passing();del record['epochs'][0]['metrics'][split]
        assert not qualified(record)
    record=passing();del record['audit']['ro_exposed'];assert not qualified(record)


@pytest.mark.parametrize("qualified",[paired_gate,policy_gate,reason_gate])
def test_completed_and_unopened_required(qualified):
    for key in ('complete','counts_pass','fresh_test_not_inferred'):
        record=passing();record[key]=False;assert not qualified(record)
    record=passing();record['selected_epoch']=2;assert not qualified(record)
