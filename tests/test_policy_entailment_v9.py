import itertools
from training.policy_entailment_v9 import targets
from roguard.evidence import ONTOLOGY,disposition


def test_factorization_preserves_every_complete_policy_world():
    names=list(ONTOLOGY)
    for values in itertools.product(*(ONTOLOGY[n] for n in names)):
        attrs=dict(zip(names,values))
        expected=int(disposition({k:dict(label=v) for k,v in attrs.items()})[0]!='CONTINUE')
        assert int(any(targets(dict(attributes=attrs))))==expected


def test_fictional_revision_is_distinct_from_personal_support():
    attrs=dict(concern='1',subject='FICTIONAL',source='FICTION',assertion='RETRACTED',relation='RETRACTS',time='CURRENT')
    assert targets(dict(attributes=attrs))==(0,1)
    attrs['assertion']='AFFIRMED';attrs['relation']='NEW'
    assert targets(dict(attributes=attrs))==(0,0)
    attrs.update(concern='0',assertion='RETRACTED',relation='RETRACTS')
    assert targets(dict(attributes=attrs))==(0,0)


def test_batch_plan_covers_both_tasks_once_and_preserves_authored_pairs():
    import random
    from training.train_policy_entailment_v9 import plan_batches
    from training.paired_scope_data_v9 import generate
    rows=generate('ro','dev')[:8]
    # Partial legacy replay groups are individual examples, not broken contrasts.
    legacy=dict(rows[0],source='legacy',pair_group='incomplete-replay')
    rows=rows+[legacy]
    batches=plan_batches(rows,random.Random(1))
    indices=[i for values,pairs in batches for i in values]
    assert sorted(indices)==list(range(2*len(rows)))
    for values,pairs in batches:
        assert len(values)<=8
        for n,p in pairs:
            assert targets(rows[values[n]//2])[values[n]%2]==0
            assert targets(rows[values[p]//2])[values[p]%2]==1
