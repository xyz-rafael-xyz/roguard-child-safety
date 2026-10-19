import pytest
from training.train_paired_nli_v9 import premise


def test_premise_excludes_future_and_marks_focus():
    row=dict(focus=1,turns=[dict(role='minor',text='before'),dict(role='minor',text='focus'),dict(role='minor',text='future')])
    text=premise(row)
    assert 'future' not in text and 'focal_message' in text and 'previous_messages' in text


def test_pair_loss_retains_neutral_mass_and_correct_direction():
    torch=pytest.importorskip('torch')
    from training.train_paired_nli_v9 import loss_function
    labels=torch.tensor([0,1])
    right=torch.tensor([[-2.,0.,2.],[2.,0.,-2.]],requires_grad=True)
    neutral=torch.tensor([[-2.,10.,2.],[2.,10.,-2.]],requires_grad=True)
    assert loss_function(right,labels,[(0,1)])<loss_function(neutral,labels,[(0,1)])
    loss_function(right,labels,[(0,1)]).backward()
    assert right.grad[0,0]>0 and right.grad[1,0]<0
