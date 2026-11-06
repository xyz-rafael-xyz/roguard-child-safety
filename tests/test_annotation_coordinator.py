from copy import deepcopy

import pytest

from eval.coordinate_annotations_v10 import coordinate
from roguard.trajectory_data import FORMAT
from roguard.trajectory_review import annotation_packet


def inputs(text='Example'):
    row = {'format': FORMAT, 'case_id': 'case', 'language': 'ro', 'turns': [{'role':'minor','text':text}],
           'split':'dev','family':'family','author':'practice','generator':'practice','synthetic':True,
           'checkpoints':[{'focus':0,'review':0,'world':None,'evidence':[]}]}
    results=[]
    for name in ('a','b'):
        packet,key=annotation_packet([row],annotator_id=name);completed=deepcopy(packet)
        completed['items'][0].update(review=0,uncertain=True,outside_ontology=False,notes='unresolved meaning')
        results.extend((packet,key,completed))
    return results


def test_disagreement_form_is_blank_and_hides_earlier_answers():
    records=inputs();records[5]['items'][0]['review']=1
    report,packet,key=coordinate(*records,adjudicator_id='third')
    assert len(report['agreement']['requires_adjudication'])==1
    assert report['gold_labels_changed'] is False
    row=packet['items'][0]
    assert row['review'] is row['world'] is row['uncertain'] is row['outside_ontology'] is None
    assert row['notes']=='' and row['evidence']==[]
    assert set(key['key'])=={row['item_id']}


def test_identical_uncertainty_still_requires_adjudication():
    report,packet,key=coordinate(*inputs(),adjudicator_id='third')
    assert len(packet['items'])==1
    assert report['independence_verified'] is False
    assert report['agreement']['jointly_resolved']==0
    assert len(report['agreement']['unresolved_checkpoints'])==1


def test_identical_resolved_returns_need_no_disagreement_packet():
    records=inputs()
    for index in (2,5):
        records[index]['items'][0].update(uncertain=False,outside_ontology=True)
    report,packet,key=coordinate(*records,adjudicator_id='third')
    assert packet is key is None
    assert report['independence_verified'] is False
    assert report['agreement']['jointly_resolved']==1


def test_matching_checkpoint_names_do_not_allow_different_inputs():
    left=inputs();right=inputs('Changed example')
    with pytest.raises(ValueError,match='same conversation'):
        coordinate(*left[:3],*right[3:],adjudicator_id='third')


def test_same_identity_cannot_fill_two_declared_roles():
    records=inputs()
    with pytest.raises(ValueError,match='Distinct'):
        coordinate(*records,adjudicator_id='a')
    with pytest.raises(ValueError,match='Distinct'):
        coordinate(*records[:3],*records[:3],adjudicator_id='third')


def test_changed_return_is_rejected_before_comparison():
    records=inputs();records[5]['items'][0]['conversation']['turns'][0]['text']='Changed'
    with pytest.raises(ValueError,match='changed'):
        coordinate(*records,adjudicator_id='third')
