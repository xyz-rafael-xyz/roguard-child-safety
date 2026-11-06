import pytest
from training.prepare_crossed_review_v10 import build,compose,SPLITS
from training.prepare_compositional_review_v10 import build as parent_build
from training.prepare_instruction_sft_v8 import target


def test_scope_crosses_preserve_policy_distinctions_without_negating_benign_cores():
    def label(assertion,subject,time,concern):
        _,attrs=compose('Un exemplu.', 'ro',assertion,subject,time,concern,0)
        return target({'attributes':attrs})
    assert label('AFFIRMED','OTHER','HYPOTHETICAL',True)==1
    assert label('NEGATED','SELF','PAST',True)==0
    assert label('NEGATED','SELF','HYPOTHETICAL',True)==0
    assert label('UNCERTAIN','SELF','CURRENT',False)==0
    assert label('UNCERTAIN','OTHER','PAST',True)==1
    with pytest.raises(ValueError,match='Negating'):compose('O activitate obișnuită.','ro','NEGATED','SELF','CURRENT',False,0)


def test_crossed_inputs_keep_parent_families_and_expose_every_registered_factor():
    rows,cores,stats=build();parent,_,_=parent_build()
    assignments={r['family']:split for split,population in parent.items() for r in population}
    assert {split:record['rows'] for split,record in stats.items()}=={'train':7680,'dev':1920,'test':3840}
    for split,population in rows.items():
        assert all(assignments[r['family']]==split and r['review']==target(r) for r in population)
        assert all(r['focus']==1 and r['turns'][0]['role']=='assistant' and r['turns'][1]['role']=='minor' for r in population)
        assert len({r['case_id'] for r in population})==len(population)
        assert len(stats[split]['factor_counts'])==30
        assert not any(r['attributes']['concern']=='0' and r['attributes']['assertion']=='NEGATED' for r in population)
    for left,right in [('train','dev'),('train','test'),('dev','test')]:
        assert not cores[left]&cores[right]
        assert not set(SPLITS[left][1])&set(SPLITS[right][1])
