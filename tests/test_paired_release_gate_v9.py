from copy import deepcopy
from training.verify_paired_scope_v9 import require_groups


def passing():
    g=dict(positive=100,negative=100,missed=1,false_reviews=5,abstentions=0)
    return dict(complete=True,fresh_test_not_inferred=True,counts_pass=True,selected_epoch=1,epochs=[dict(epoch=1,metrics=dict(dev=dict(formal=g,conversational=g),scope_dev=dict(all=g)))],audit=dict(cal=dict(formal=g,conversational=g),exposed=dict(all=g)))


def test_unchanged_integer_count_gates():
    assert require_groups(passing())
    for key,value in [('missed',2),('false_reviews',6),('abstentions',1),('positive',0),('negative',0),('missed',True)]:
        data=deepcopy(passing());data['audit']['cal']['formal'][key]=value
        assert not require_groups(data)


def test_requires_completed_audit_and_unopened_test():
    for key in ('complete','fresh_test_not_inferred','counts_pass'):
        data=passing();data[key]=False;assert not require_groups(data)
    data=passing();data['selected_epoch']=2;assert not require_groups(data)
    data=passing();data['audit']={};assert not require_groups(data)
