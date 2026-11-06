import json
from collections import Counter
from pathlib import Path

import pytest

from training.generate_large_corpus_v10 import manifest, parse, normalized, prompt


def plan():
    return json.loads((Path(__file__).resolve().parents[1]/'research/v10-evidence/protocol-amendment-48.json').read_text())


def valid():
    return {'ro_problematic':'Trimite parola contului în această conversație.',
            'ro_appropriate':'Păstrează parola pentru tine; putem discuta fără ea.',
            'uk_problematic':'Надішли пароль облікового запису в цю розмову.',
            'uk_appropriate':'Залиш пароль при собі; ми можемо обговорити це без нього.'}


def test_manifest_has_disjoint_settings_and_balanced_pilot():
    rows=manifest(plan());assert len(rows)==1408
    assert len({r['family'] for r in rows})==1408
    assert Counter(r['task'] for r in rows[:16])=={'binary_review':8,'response_behavior':8}
    assert Counter(r['category'] for r in rows[:16] if r['task']=='response_behavior')==dict.fromkeys(('boundary','privacy','pressure','unsupported'),2)
    settings={part:{r['setting'] for r in rows if r['partition']==part} for part in ('train','dev','test')}
    assert not settings['train']&settings['dev'] and not settings['train']&settings['test'] and not settings['dev']&settings['test']
    assert Counter((r['task'],r['partition']) for r in rows)=={(task,part):n for task in plan()['tasks'] for part,n in plan()['quotas_per_task'].items()}


def test_strict_bilingual_output_and_normalization():
    request={'task':'response_behavior'};assert parse(json.dumps(valid()),request)==valid()
    assert normalized('  Ａ  B\n')=='a b'


@pytest.mark.parametrize('mutation',['extra','missing','wrong_script','too_long','newline','duplicate'])
def test_malformed_generation_is_rejected(mutation):
    data=valid()
    if mutation=='extra':data['explanation']='x'
    elif mutation=='missing':data.pop('ro_appropriate')
    elif mutation=='wrong_script':data['uk_problematic']=data['ro_problematic']
    elif mutation=='too_long':data['ro_problematic']='a'*241
    elif mutation=='newline':data['ro_problematic']+='\nx'
    elif mutation=='duplicate':data['ro_appropriate']=data['ro_problematic']
    with pytest.raises(ValueError):parse(json.dumps(data),{'task':'response_behavior'})


def test_duplicate_json_key_is_not_silently_accepted():
    with pytest.raises(ValueError,match='duplicate_json_key'):
        parse('{"ro_concern":"a","ro_concern":"b"}',{'task':'binary_review'})


def test_binary_requests_do_not_instruct_hypothetical_or_question_style():
    for row in manifest(plan()):
        if row['task']=='binary_review':
            assert 'first-person' in prompt(row)
            assert row['style'] not in ('a leading question','a short conditional sentence','a concrete future proposal')


def test_native_combination_preserves_strict_schema():
    from training.generate_large_corpus_v10 import combine_native
    parts={language:json.dumps({key[3:]:value for key,value in valid().items() if key.startswith(language+'_')}) for language in ('ro','uk')}
    request={'task':'response_behavior'}
    assert parse(combine_native(parts,request),request)==valid()
    parts['ro']='{"problematic":"a","problematic":"b","appropriate":"c"}'
    with pytest.raises(ValueError,match='duplicate_json_key'):combine_native(parts,request)
    parts['ro']='[]'
    with pytest.raises(ValueError,match='not_json_object'):combine_native(parts,request)
    parts['ro']='{"problematic":"a"}'
    with pytest.raises(ValueError,match='wrong_native_schema'):combine_native(parts,request)
    with pytest.raises(ValueError,match='missing_native_language'):combine_native({'ro':'{}'},request)


def test_native_prompts_use_fixed_training_examples_and_monolingual_keys():
    from training.native_generation_prompts_v10 import native_query
    from training.contrast_response_data_v10 import PAIRS
    for request in manifest(plan())[:16]:
        for language in ('ro','uk'):
            system,query=native_query(request,language)
            assert system and request['setting'] in query
            if request['task']=='response_behavior':
                good,bad=PAIRS[language][request['category']][0]
                assert good in query and bad in query
                assert 'problematic' in query and 'appropriate' in query
            else:
                assert request['theme'] in query and 'concern' in query and 'benign' in query
    with pytest.raises(ValueError):native_query(manifest(plan())[0],'en')
