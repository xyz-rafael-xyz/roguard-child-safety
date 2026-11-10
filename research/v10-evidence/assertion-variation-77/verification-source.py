"""Independently recount matched TRAIN rewrites and retain exact runtime inputs."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import unicodedata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private', 'base', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite verification')
    from roguard.assurance import digest
    from roguard.evidence import ONTOLOGY
    from roguard.gemma4_contract import BASE_REVISION, FILES, PREFIX_TOKENS, classifier_tokens
    from roguard.trajectory_data import file_sha, prefix, read_jsonl
    from training.evidence_v8 import WORDS, ASSERTIONS
    from training.prepare_assertion_variation_v10 import OPERATORS  # Wording data only.
    from transformers import AutoTokenizer

    study = a.root/'research/v10-evidence/assertion-variation-77'
    plan_path = a.root/'research/v10-evidence/assertion-variation-plan-77.json'
    plan = json.loads(plan_path.read_text()); report = json.loads((study/'report.json').read_text())
    assert report['complete'] and report['plan_sha256'] == file_sha(plan_path)
    assert report['source_sha256'] == file_sha(study/'source.py') == file_sha(a.root/'training/prepare_assertion_variation_v10.py')
    for base, bindings in ((a.root, plan['public']), (a.private, plan['private'])):
        for name, expected in bindings.items():
            assert file_sha(base/name) == expected, name
    data = a.private/'assertion-variation-77/actual'
    for name, expected in report['artifacts'].items():
        assert file_sha(data/name) == expected, name
    assert a.base.name == BASE_REVISION
    tokenizer_files = {name: file_sha(a.base/name) for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}
    for name, actual in tokenizer_files.items():
        assert actual == FILES[name]['sha256']
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)

    def normalize(text):
        return ' '.join(unicodedata.normalize('NFC', text).casefold().split())

    def marker(text):
        return bool(re.search(r'\b(?:nu|не)\b', normalize(text)))

    originals = {}
    for name in plan['historical_sources']:
        with gzip.open(a.root/name, 'rt') as stream:
            for line in stream:
                row = json.loads(line)['row']; key = (row['language'], row['case_id'])
                assert key not in originals
                originals[key] = row
    recovered = {}; protected_texts = set(); protected_tokens = set(); choices = {}; eligible = Counter()
    with (a.private/plan['recovered_inputs']).open() as stream:
        for line in stream:
            entry = json.loads(line); key = (entry['language'], entry['case_id'])
            assert key not in recovered
            recovered[key] = entry
            protected_texts.add(normalize(entry['payload']['turns'][-1]['text']))
            protected_tokens.add((entry['language'], entry['token_sha256']))
            old = originals.get(key)
            if entry['partition'] != 'train' or old is None or old['source'] != 'controlled_composition_v8':
                continue
            assert old['case_id'].startswith('train-') and digest(old) == entry['source_row_sha256']
            assert prefix(old, old['focus']) == entry['payload']
            assert [ONTOLOGY[k].index(old['attributes'][k]) for k in ONTOLOGY] == entry['fields']
            text = old['turns'][old['focus']]['text']
            assert text.count('«') == text.count('»') == 1
            core = text.split('«')[1].split('»')[0]
            choice = (entry['language'], normalize(core), tuple(entry['fields']))
            eligible[entry['language']] += 1
            previous = choices.get(choice)
            if previous is None or digest(old) < digest(originals[previous]):
                choices[choice] = key
    parents = read_jsonl(data/'parents.jsonl')
    assert len(parents) == len(choices) == report['selected_parents']
    assert dict(eligible) == report['eligible_original_train_rows']
    assert [(p['language'], p['case_id']) for p in parents] == [choices[k] for k in sorted(choices)]
    parent_ids = {}
    for p in parents:
        key = (p['language'], p['case_id']); old = originals[key]; entry = recovered[key]
        core = old['turns'][old['focus']]['text'].split('«')[1].split('»')[0]
        expected = {'language': old['language'], 'case_id': old['case_id'], 'source_row_sha256': digest(old),
                    'feature_index': entry['feature_index'], 'token_sha256': entry['token_sha256'],
                    'fields': entry['fields'], 'gold': entry['gold'], 'core_sha256': digest(normalize(core)),
                    'family': old.get('family'), 'ordinary_family': old.get('ordinary_family')}
        assert p == expected
        parent_ids[key] = p

    treatment = read_jsonl(data/'additional-train.jsonl'); control = read_jsonl(data/'matched-control.jsonl')
    assert len(treatment) == len(control) == len(parents)*8 == report['rows_per_arm']
    a.output.mkdir(parents=True)
    cases = set(); tokens_seen = set(); counts = Counter(); parent_factors = {}; descriptors = Counter()
    lengths = []; control_token_cache = {}; treatment_by_language = Counter()
    destination = a.output/'runtime-inputs.jsonl'
    with destination.open('x') as stream:
        for changed, repeated in zip(treatment, control, strict=True):
            key = (changed['parent']['language'], changed['parent']['case_id'])
            p = parent_ids[key]; old = originals[key]; entry = recovered[key]
            factors = changed['factors']; op = factors['operator_index']; placement = factors['quote_placement']
            assert op in range(4) and placement in ('before', 'after')
            assert factors == {'operator_index': op, 'quote_placement': placement, 'operator_negation_marker': op >= 2}
            parent_factors.setdefault(key, []).append((op, placement))
            pair_id = 'assertion77:'+p['language']+':'+digest(p)[:20]+':'+str(op)+':'+placement
            for arm, row in (('treatment', changed), ('control', repeated)):
                assert row['case_id'] == pair_id+':'+arm and row['case_id'] not in cases
                cases.add(row['case_id'])
                assert row['pair_id'] == pair_id and row['arm'] == arm
                assert row['partition'] == 'train' and row['parent'] == p and row['factors'] == factors
                assert row['review'] == entry['gold'] and row['attributes'] == old['attributes']
                assert row['synthetic_only'] is True and row['independently_annotated'] is False
                for field, value in old.items():
                    if field not in ('case_id', 'turns'):
                        assert row[field] == value
                assert len(row['turns']) == len(old['turns'])
                for i, turn in enumerate(old['turns']):
                    if i != old['focus']:
                        assert row['turns'][i] == turn
                assert row['turns'][row['focus']]['role'] == old['turns'][old['focus']]['role']
                actual_text = row['turns'][row['focus']]['text']
                assert row['literal_pre_quote_negation'] == marker(actual_text.split('«')[0])
            assert repeated['turns'] == old['turns']
            text = old['turns'][old['focus']]['text']; new_text = changed['turns'][changed['focus']]['text']
            left, core_and_tail = text.split('«'); core, tail = core_and_tail.split('»')
            old_operator = WORDS[old['language']]['assertion'][0][ASSERTIONS.index(old['attributes']['assertion'])]
            assert left.endswith(old_operator+' ')
            surrounding_left = left[:-len(old_operator+' ')]
            operator = OPERATORS[old['language']][old['attributes']['assertion']][op]
            assert marker(operator) == (op >= 2)
            quote = '«'+core+'»'
            expected = surrounding_left+(operator+' '+quote if placement == 'before' else quote+' '+operator)+tail
            assert new_text == expected and normalize(new_text) not in protected_texts
            assert changed['literal_pre_quote_negation'] == (op >= 2 and placement == 'before')
            payload = prefix(changed, changed['focus']); tokens = classifier_tokens(tok, payload)
            assert tokens[:len(PREFIX_TOKENS)] == PREFIX_TOKENS and 0 < len(tokens) <= 1024
            identity = (changed['language'], digest(tokens))
            assert identity not in tokens_seen and identity not in protected_tokens
            tokens_seen.add(identity); lengths.append(len(tokens)); treatment_by_language[changed['language']] += 1
            if key not in control_token_cache:
                reference = classifier_tokens(tok, prefix(repeated, repeated['focus']))
                assert reference == entry['tokens'] and digest(reference) == entry['token_sha256']
                control_token_cache[key] = digest(reference)
            count_key = (changed['language'], changed['attributes']['assertion'], op >= 2, placement, changed['review'])
            counts[count_key] += 1
            descriptors[(changed['language'], changed['attributes']['assertion'], changed['literal_pre_quote_negation'])] += 1
            record = {'case_id': changed['case_id'], 'language': changed['language'], 'partition': 'train',
                      'pair_id': pair_id, 'parent': p, 'factors': factors, 'fields': entry['fields'], 'gold': entry['gold'],
                      'payload': payload, 'tokens': tokens, 'token_count': len(tokens), 'token_sha256': digest(tokens)}
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    assert all(sorted(values) == [(i, pos) for i in range(4) for pos in ('after', 'before')] for values in parent_factors.values())
    expected_counts = [{'language': k[0], 'assertion': k[1], 'operator_negation_marker': k[2],
                        'quote_placement': k[3], 'review': k[4], 'rows': n} for k, n in sorted(counts.items())]
    assert expected_counts == report['counts']
    for lang in ('ro', 'uk'):
        for assertion in ASSERTIONS:
            assert descriptors[(lang, assertion, False)] == 3*descriptors[(lang, assertion, True)]
        assert len({p['core_sha256'] for p in parents if p['language'] == lang}) == report['unique_cores_by_language'][lang]
    result = {'complete': True, 'verified': True, 'source_sha256': file_sha(__file__),
              'plan_sha256': file_sha(plan_path), 'data_report_sha256': file_sha(study/'report.json'),
              'parents_verified': len(parents), 'treatment_rows_verified': len(treatment),
              'control_rows_verified': len(control), 'treatment_rows_by_language': dict(treatment_by_language),
              'runtime_inputs_sha256': file_sha(destination), 'runtime_inputs_bytes': destination.stat().st_size,
              'token_count': {'minimum': min(lengths), 'maximum': max(lengths)},
              'unique_control_tokens_reproduced': len(control_token_cache), 'tokenizer_files': tokenizer_files,
              'base_revision': BASE_REVISION, 'unchanged_context_core_labels_verified': True,
              'all_parent_choices_minimum_source_digest': True, 'all_factor_cells_and_report_counts_reproduced': True,
              'pre_quote_marker_rate_per_language_assertion': 0.25,
              'runtime_token_overlap_with_existing_train_dev': 0, 'runtime_prefix_verified': True,
              'encoder_loaded': False, 'training_performed': False, 'training_admitted': False,
              'independent_language_or_label_validation': False, 'reserved_test_opened': False, 'qualified': False}
    (a.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
