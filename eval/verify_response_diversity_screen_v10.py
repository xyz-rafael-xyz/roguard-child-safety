"""Independently recount screen81 raw judgments; never certify their semantic truth."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl

LABELS = ('appropriate', 'boundary', 'privacy', 'pressure', 'unsupported')
CODES = dict(zip('ABCDE', LABELS))
PLAN_SHA = 'd9d455ca40815f6d9e71e75057fe343945dd2f8e29bc7f9b86a0aa1dc6db67be'
QUEUE_SHA = '620a41f9c0d63386a8e0d0130f61eedd0f3bf852e9874d9e2b12735db2b4d9da'
SCOPE = dict.fromkeys(('training_ready', 'independent_annotation', 'semantic_labels_verified',
                      'candidate_training_changed', 'held_out_test_opened', 'stable_release_ready'), False)


def require(value, message):
    if not value:
        raise ValueError(message)


def equal(actual, expected, message):
    # Python equality alone equates True with 1, including nested records.
    require(digest(actual) == digest(expected), message)


def decode_judgment(raw):
    """Separate parser: do not call the screen's parse/result/recount helpers."""
    def object_pairs(pairs):
        keys = [key for key, _ in pairs]
        require(len(keys) == len(set(keys)), 'Duplicate JSON key')
        return dict(pairs)

    require(type(raw) is str, 'Raw generation is not text')
    try:
        value = json.loads(raw, object_pairs_hook=object_pairs)
        require(type(value) is dict and set(value) ==
                {'label', 'language_ok', 'label_reason', 'language_reason'}, 'Judgment schema')
        require(type(value['label']) is str and value['label'] in (*CODES, 'unclear'), 'Label')
        require(type(value['language_ok']) is bool, 'Language flag')
        for key in ('label_reason', 'language_reason'):
            require(type(value[key]) is str and 1 <= len(value[key].strip()) <= 2000, 'Reason')
    except (ValueError, TypeError):
        return None
    return value


def ordered_rows(rows):
    require(len(rows) == 128 and len({r['case_id'] for r in rows}) == 128, 'Draft population')
    for row in rows:
        require(row['partition'] == 'train' and row['context'] == ''
                and row['independently_annotated'] is False
                and row['language'] in ('ro', 'uk') and row['gold'] in LABELS
                and type(row['draft']) is str and bool(row['draft'].strip()), 'Draft scope')
    equal(dict(Counter(r['language']+':'+r['gold'] for r in rows)),
          {lang+':'+label: 32 if label == 'appropriate' else 8
           for lang in ('ro', 'uk') for label in LABELS}, 'Draft balance')
    families = {r['family'] for r in rows}
    require(len(families) == 32, 'Family count')
    for family in families:
        members = [r for r in rows if r['family'] == family]
        categories = {r['gold'] for r in members} - {'appropriate'}
        require(len(members) == 4 and len(categories) == 1, 'Family members')
        require(Counter((r['language'], r['gold']) for r in members) ==
                Counter((lang, label) for lang in ('ro', 'uk')
                        for label in ('appropriate', *categories)), 'Bilingual family coverage')
    return sorted(rows, key=lambda r: (digest({'study': 81, 'case_id': r['case_id']}), r['case_id']))


def replay_tokens(rows, screen_prompt, tokenizer):
    tokens = {}
    for row in rows:
        payload = {'language': {'ro': 'Romanian', 'uk': 'Ukrainian'}[row['language']],
                   'prior_turns': [], 'focal_message': {'role': 'assistant', 'text': row['draft']}}
        messages = [{'role': 'system', 'content': screen_prompt},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        value = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                              add_generation_prompt=True, enable_thinking=False)
        require(isinstance(value, list) and 0 < len(value) <= 2048
                and all(type(t) is int and t >= 0 for t in value), 'Query token envelope')
        tokens[row['case_id']] = value
    return tokens


def recount_raw(rows, records, tokens):
    rows = ordered_rows(rows)
    require([r['case_id'] for r in records] == [r['case_id'] for r in rows], 'Judgment population or order')
    require(set(tokens) == {r['case_id'] for r in rows}, 'Token population')
    counts = dict(rows=128, accepted_rows=0, malformed_outputs=0, label_disagreements=0, language_flags=0)
    groups = {lang+':'+label: {'rows': 0, 'accepted': 0} for lang in ('ro', 'uk') for label in LABELS}
    families = {r['family']: True for r in rows}
    for row, record in zip(rows, records):
        require(set(record) == {'case_id', 'input_sha256', 'query_token_sha256', 'raw',
                                'judgment', 'parse_error', 'accepted'}, 'Retained record schema')
        judgment = decode_judgment(record['raw'])
        equal(record['judgment'], judgment, 'Retained judgment differs from raw')
        malformed = judgment is None
        disagree = not malformed and CODES.get(judgment['label']) != row['gold']
        language_flag = not malformed and judgment['language_ok'] is False
        accepted = not (malformed or disagree or language_flag)
        equal(record['accepted'], accepted, 'Retained acceptance differs')
        if malformed:
            require(type(record['parse_error']) is str and 0 < len(record['parse_error']) <= 160,
                    'Missing malformed-output diagnostic')
        else:
            require(record['parse_error'] is None, 'Unexpected parse diagnostic')
        require(record['input_sha256'] == digest(row)
                and record['query_token_sha256'] == digest(tokens[row['case_id']]), 'Input or token identity')
        for key, increment in (('accepted_rows', accepted), ('malformed_outputs', malformed),
                               ('label_disagreements', disagree), ('language_flags', language_flag)):
            counts[key] += int(increment)
        group = groups[row['language']+':'+row['gold']]
        group['rows'] += 1
        group['accepted'] += int(accepted)
        families[row['family']] &= accepted
    return {**counts, 'groups': groups, 'family_pass': dict(sorted(families.items())),
            'accepted_families': sum(families.values()),
            'decision': ('MACHINE_SCREEN_PASS_REQUIRES_SEPARATE_FIT_REGISTRATION'
                         if counts['accepted_rows'] == 128 else 'REJECT_FOR_FITTING_UNDER_THIS_SCREEN')}


def check_sequence(sequence, queue, queue_sha):
    require(sequence.get('execution_complete') is True and sequence.get('phase') == 'finished',
            'Screen sequence is incomplete')
    require(sequence['plan_sha256'] == queue_sha, 'Queue plan changed')
    equal(sequence['source_hashes'], queue['source_hashes'], 'Queue bindings differ')
    require([s['name'] for s in sequence['completed_steps']] == [s['name'] for s in queue['steps']]
            and all(type(s['exit_code']) is int and s['exit_code'] == 0 for s in sequence['completed_steps']),
            'Screen sequence failed or omitted a step')


def verify_report(report, protocol, progress, counts):
    require(report.get('complete') is True, 'Screen incomplete')
    for value in (report, protocol, progress):
        for key in SCOPE:
            require(value.get(key) is False, 'Screen scope was overstated: '+key)
    for key, value in counts.items():
        equal(report[key], value, 'Reported recount differs: '+key)
    require(type(progress['completed_rows']) is int and progress['completed_rows'] == 128, 'Progress incomplete')
    require(all(type(v['seconds']) in (float, int) and math.isfinite(v['seconds']) and v['seconds'] >= 0
                for v in (progress, report)) and progress['seconds'] <= report['seconds'], 'Invalid timing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private', 'base', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    require(not a.output.exists(), 'Refusing to overwrite verification')
    evidence = a.root/'research/v10-evidence'
    plan_path = evidence/'protocol-amendment-70.json'
    queue_path = a.private/'response-diversity-screen-sequence-81-plan.json'
    require(file_sha(plan_path) == PLAN_SHA and file_sha(queue_path) == QUEUE_SHA, 'Frozen plan changed')
    plan = json.loads(plan_path.read_text()); queue = json.loads(queue_path.read_text())
    sequence_path = evidence/'response-diversity-screen-sequence-81/report.json'
    sequence = json.loads(sequence_path.read_text())
    check_sequence(sequence, queue, QUEUE_SHA)
    require(sequence['source_sha256'] == file_sha(a.root/'scripts/run_report_sequence_v10.py')
            and sequence['dependency_report_sha256'] == file_sha(a.root/plan['dependency_report'])
            and sequence['stable_release_ready'] is False and sequence['promotion_performed'] is False,
            'Sequence provenance or scope differs')
    pid = int((a.private/'response-diversity-screen-sequence-81.pid').read_text())
    require(pid == 10514, 'Registered queue PID differs')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Screen queue has not exited')
    for root, bindings in ((a.root, plan['public']), (a.private, plan['private']),
                           (a.root, queue['source_hashes'])):
        for name, expected in bindings.items():
            require(file_sha(root/name) == expected, 'Bound input changed: '+name)
    study = evidence/'response-diversity-screen-81'
    protocol = json.loads((study/'protocol.json').read_text())
    report = json.loads((study/'report.json').read_text())
    progress = json.loads((study/'progress.json').read_text())
    source_sha = plan['public']['training/screen_response_diversity_v10.py']
    require(file_sha(study/'source.py') == source_sha and protocol['source_sha256'] == source_sha
            and protocol['plan_sha256'] == PLAN_SHA, 'Protocol source binding')
    for key in ('judge', 'model_files', 'software', 'public', 'private', 'task_prompt_sha256'):
        equal(protocol[key], plan[key], 'Protocol differs: '+key)
    require(protocol['format'] == 'roguard-v10-single-response-screen-1'
            and digest(protocol['screen_prompt']) == plan['screen_prompt_sha256'], 'Screen prompt differs')
    # Tokenizer-only replay: never import/load a model or regenerate judgments.
    require(a.base.name == plan['judge']['revision'], 'Tokenizer revision differs')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json'):
        require(file_sha(a.base/name) == plan['model_files'][name], 'Tokenizer file changed: '+name)
    for name in ('transformers', 'tokenizers'):
        require(version(name) == plan['software'][name], 'Tokenizer software differs: '+name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.base), local_files_only=True, trust_remote_code=False)
    rows_path = a.private/'response-diversity-pilot-69/train-draft.jsonl'
    rows = ordered_rows(read_jsonl(rows_path))
    tokens = replay_tokens(rows, protocol['screen_prompt'], tokenizer)
    for key, value in {'rows': 128, 'ordered_inputs_sha256': digest(rows), 'query_tokens_sha256': digest(tokens),
                       'minimum_query_tokens': min(map(len, tokens.values())),
                       'maximum_query_tokens': max(map(len, tokens.values()))}.items():
        equal(protocol[key], value, 'Replayed inputs differ: '+key)
    raw_path = a.private/'response-diversity-screen-81/judgments.jsonl'
    require(report['judgments_sha256'] == file_sha(raw_path)
            and report['protocol_sha256'] == file_sha(study/'protocol.json'), 'Output file binding differs')
    counts = recount_raw(rows, read_jsonl(raw_path), tokens)
    verify_report(report, protocol, progress, counts)
    proof = {'verification': 'PASS', 'source_sha256': file_sha(__file__), 'plan_sha256': PLAN_SHA,
             'queue_plan_sha256': QUEUE_SHA, 'sequence_sha256': file_sha(sequence_path),
             'report_sha256': file_sha(study/'report.json'), 'protocol_sha256': file_sha(study/'protocol.json'),
             'progress_sha256': file_sha(study/'progress.json'), 'judgments_sha256': file_sha(raw_path),
             'drafts_sha256': file_sha(rows_path), 'query_tokens_sha256': digest(tokens), 'recount': counts,
             'model_loaded': False, 'generations_repeated': False, 'qualified': False, **SCOPE,
             'limitations': 'Independent software recount and tokenizer replay only. Raw generations are retained '
                            'records, not repeated inference. Machine agreement does not verify language or labels. '
                            'A rejected screen can have a passing integrity audit. No fitting admission.'}
    with a.output.open('x') as output:
        output.write(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
