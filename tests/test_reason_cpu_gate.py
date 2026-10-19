from copy import deepcopy
from training.verify_reason_cpu_v9 import ready
from test_paired_nli_release_gate import passing


def installed():
    count=dict(positive=150,negative=150,missed=1,false_reviews=7,abstentions=0)
    return dict(complete=True,parity_pass=True,prerequisite_accuracy_pass=True,
                fresh_test_inferred=True,fresh_counts_pass=True,
                fresh={lang:dict(counts=count) for lang in ('ro','uk')})


def test_full_cpu_requires_all_prior_evidence():
    assert ready(passing(), installed())
    for key in ('complete','parity_pass','prerequisite_accuracy_pass','fresh_test_inferred','fresh_counts_pass'):
        actual=installed();actual[key]=False
        assert not ready(passing(), actual)
    parent=passing();parent['counts_pass']=False
    assert not ready(parent, installed())


def test_fresh_flags_cannot_override_failing_or_incomplete_counts():
    for key,value in [('positive',149),('negative',0),('missed',2),('false_reviews',8),('abstentions',1),('missed',True)]:
        actual=deepcopy(installed());actual['fresh']['uk']['counts'][key]=value
        assert not ready(passing(), actual)
    actual=installed();del actual['fresh']['ro']
    assert not ready(passing(), actual)
