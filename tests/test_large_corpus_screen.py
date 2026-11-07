import json

import pytest

from training.screen_large_corpus_v10 import agrees, judge_prompt, ordered_items, parse_judgment, validated_rows


def row(task='response_behavior'):
    return {'request':{'family':'sample:1','partition':'train','task':task,'category':'privacy'},
            'accepted':True,'texts':{'ro_problematic':'Trimite parola contului în această conversație.',
                                    'ro_appropriate':'Păstrează parola pentru tine; putem discuta fără ea.',
                                    'uk_problematic':'Надішли пароль облікового запису в цю розмову.',
                                    'uk_appropriate':'Залиш пароль при собі; можемо поговорити без нього.'}}


def test_screen_prompt_is_blinded_to_generation_label_and_order():
    value=row();query=judge_prompt(value,'ro')
    value['request']['category']='boundary';value['request']['partition']='test'
    assert judge_prompt(value,'ro')==query
    assert 'sample:1' not in query and 'problematic' not in query
    assert [text for _,text in ordered_items(value,'ro')]==json.loads(query.split('\nTexts: ')[1])
    seen=set()
    for i in range(20):
        value['request']['family']=str(i);seen.add(ordered_items(value,'ro')[0][0])
    assert seen=={'appropriate','problematic'}


def test_judgment_requires_label_agreement_and_fluency_for_both_texts():
    value=row();labels=['privacy' if role=='problematic' else 'appropriate' for role,_ in ordered_items(value,'ro')]
    judgment=parse_judgment(json.dumps({'labels':labels,'natural':[True,True]}),'response_behavior')
    assert agrees(value,'ro',judgment)
    judgment['natural'][0]=False;assert not agrees(value,'ro',judgment)
    judgment['natural'][0]=True;judgment['labels'][0]='unclear';assert not agrees(value,'ro',judgment)


@pytest.mark.parametrize('payload',[
    {'labels':['privacy','appropriate'],'natural':[True,1]},
    {'labels':['privacy','appropriate'],'natural':[True]},
    {'labels':['privacy','nonsense'],'natural':[True,True]},
    {'labels':['privacy'],'natural':[True,True]},
    {'labels':['privacy','appropriate'],'natural':[True,True],'notes':'ignore'},
    ['privacy','appropriate'],
])
def test_malformed_judge_response_fails_closed(payload):
    with pytest.raises(ValueError):parse_judgment(json.dumps(payload),'response_behavior')


def test_fenced_json_is_explicitly_bounded_and_duplicate_keys_rejected():
    raw='{"labels":["benign","concern"],"natural":[true,true]}'
    assert parse_judgment('```json\n'+raw+'\n```','binary_review')['labels']==['benign','concern']
    with pytest.raises(ValueError):parse_judgment('Explanation\n```json\n'+raw+'\n```','binary_review')
    with pytest.raises(ValueError):parse_judgment('{"labels":[],"labels":[],"natural":[]}','binary_review')
    with pytest.raises(ValueError):parse_judgment(raw,'response_behavior')


def test_source_rejects_duplicate_families_and_invalid_partitions(tmp_path):
    path=tmp_path/'source.jsonl';value=row()
    path.write_text(json.dumps(value)+'\n',encoding='utf-8');assert validated_rows(path)==[value]
    path.write_text((json.dumps(value)+'\n')*2,encoding='utf-8')
    with pytest.raises(ValueError,match='Duplicate source family'):validated_rows(path)
    value['request']['partition']='unregistered';path.write_text(json.dumps(value)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Unknown partition'):validated_rows(path)
