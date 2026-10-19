from copy import deepcopy
import pytest
from training.verify_stress_bilingual_v9 import validate_stress_development


def selection():
    group = dict(rows=216, positive=44, negative=172, missed=0, false_reviews=8, abstentions=0)
    return {'stress_development': group, 'stress_oof': deepcopy(group)}


def test_full_stress_gate_accepts_exact_boundary():
    validate_stress_development(selection())


@pytest.mark.parametrize('key,value', [('missed',1),('false_reviews',9),('rows',215),('positive',45),
                                     ('abstentions',1),('missed',False),('negative',None)])
def test_family_validation_or_refit_failure_blocks_reserved_access(key,value):
    for name in ('stress_development','stress_oof'):
        record=selection();record[name][key]=value
        with pytest.raises(ValueError):validate_stress_development(record)


def test_missing_fold_results_block_reserved_access():
    record=selection();del record['stress_oof']
    with pytest.raises(ValueError):validate_stress_development(record)
