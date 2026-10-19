from copy import deepcopy
import pytest
from training.verify_native_coverage_v9 import require_groups
from training.verify_native_morphology_v9 import require_groups as morphology_gate
from training.verify_native_scope_operators_v9 import require_groups as operators_gate


def passing(language):
    g=dict(positive=100,negative=100,missed=1,false_reviews=5,abstentions=0)
    groups={'dev':dict(formal=g,conversational=g),'scope_dev':dict(all=g)}
    if language=='uk':groups['stress']=dict(all=g)
    return dict(language=language,complete=True,fresh_test_not_inferred=True,counts_pass=True,
                selected_epoch=1,epochs=[dict(epoch=1,metrics=groups)],
                audit=dict(cal=dict(formal=g,conversational=g),exposed=dict(all=g)))


@pytest.mark.parametrize('language',['ro','uk'])
@pytest.mark.parametrize('gate',[require_groups,morphology_gate,operators_gate])
def test_requires_every_group_and_unchanged_accuracy_limits(language,gate):
    assert gate(passing(language))
    for name in passing(language)['epochs'][0]['metrics']:
        r=deepcopy(passing(language));del r['epochs'][0]['metrics'][name]
        assert not gate(r)
    for key,value in [('missed',2),('false_reviews',6),('abstentions',1),('positive',0),('negative',0),('missed',True)]:
        r=deepcopy(passing(language));r['audit']['cal']['conversational'][key]=value
        assert not gate(r)
    r=deepcopy(passing(language));del r['audit']['cal']['formal']
    assert not gate(r)


@pytest.mark.parametrize('gate',[require_groups,morphology_gate,operators_gate])
def test_rejects_unfinished_or_previously_opened_candidates(gate):
    for key in ('complete','fresh_test_not_inferred','counts_pass'):
        r=passing('uk');r[key]=False
        assert not gate(r)
    r=passing('uk');r['selected_epoch']=3
    assert not gate(r)
    r=passing('uk');r['language']='en'
    assert not gate(r)
