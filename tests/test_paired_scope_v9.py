from collections import defaultdict
from training.paired_scope_data_v9 import generate,CONTENT,FRAMES
from training.train_bare_distress_v9 import texts
from training.prepare_instruction_sft_v8 import target


def test_partitions_have_disjoint_content_frames_and_inputs():
    for language in ('ro','uk'):
        keys={split:{texts(r) for r in generate(language,split)} for split in ('train','dev','test')}
        for a,b in [('train','dev'),('train','test'),('dev','test')]:
            assert not keys[a]&keys[b]
            assert not set(CONTENT[language][a])&set(CONTENT[language][b])
            af={v for values in FRAMES[language][a].values() for pair in values for v in pair}
            bf={v for values in FRAMES[language][b].values() for pair in values for v in pair}
            assert not af&bf


def test_counterfactual_pairs_match_policy_and_are_complete():
    for language in ('ro','uk'):
        for split in ('train','dev','test'):
            rows=generate(language,split);pairs=defaultdict(list)
            assert len(rows)==len({texts(r) for r in rows})
            for row in rows:
                assert row['review']==target(row)
                assert row['focus']==len(row['turns'])-1
                if row['pair_group']:pairs[row['pair_group']].append(row)
            for values in pairs.values():
                assert sorted(r['review'] for r in values)==[0,1]
                assert len({r['content_family'] for r in values})==1


def test_paired_loss_penalizes_reversed_counterfactual_order():
    import pytest
    torch=pytest.importorskip('torch')
    from training.train_paired_scope_v9 import paired_loss
    labels=torch.tensor([0,1])
    good=torch.tensor([[2.,-2.],[-2.,2.]],requires_grad=True)
    bad=torch.tensor([[-2.,2.],[2.,-2.]],requires_grad=True)
    assert paired_loss(good,labels,[(0,1)])<paired_loss(bad,labels,[(0,1)])
    loss=paired_loss(bad,labels,[(0,1)]);loss.backward()
    assert bad.grad[0,1]>0 and bad.grad[1,1]<0
