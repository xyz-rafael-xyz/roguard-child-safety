"""Recompute the authored pilot's machine-screen results without model inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_raw(raw):
    text = raw.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4].strip()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique)
    allowed = {'appropriate', 'boundary', 'privacy', 'pressure', 'unsupported', 'unclear'}
    if not isinstance(value, dict) or set(value) != {'labels', 'natural'}:
        raise ValueError('Wrong schema')
    if not isinstance(value['labels'], list) or len(value['labels']) != 2:
        raise ValueError('Wrong label count')
    if any(not isinstance(x, str) or x not in allowed for x in value['labels']):
        raise ValueError('Invalid label')
    if not isinstance(value['natural'], list) or len(value['natural']) != 2:
        raise ValueError('Wrong fluency count')
    if any(type(x) is not bool for x in value['natural']):
        raise ValueError('Invalid fluency flag')
    return value


def verify(root, private):
    sys.path[:0] = [str(root / 'src'), str(root)]
    from roguard.assurance import digest
    ev = root / 'research/v10-evidence'
    study = ev / 'authored-response-screen-66'
    rawdir = private / 'authored-response-screen-66'
    source = private / 'authored-response-pilot-66/families-revision1.jsonl'
    amendment = ev / 'protocol-amendment-64.json'
    plan = json.loads(amendment.read_text())
    report = json.loads((study / 'report.json').read_text())
    protocol = json.loads((study / 'protocol.json').read_text())
    preflight = json.loads((ev / 'authored-response-screen-preflight-66/report.json').read_text())
    assert report['complete'] is True
    for key in ('training_ready', 'semantic_labels_verified', 'independent_annotation', 'stable_release_ready'):
        assert report[key] is False
    for path, expected in plan['bindings'].items():
        assert sha(root / path) == expected, path
    assert sha(amendment) == preflight['amendment_sha256'] == protocol['amendment_sha256']
    assert sha(source) == protocol['corpus_sha256'] == preflight['corpus_sha256']
    assert protocol['judge'] == plan['judge']
    assert protocol['model_files'] == plan['model_files'] == preflight['model_files']
    assert protocol['source_sha256'] == sha(root / 'training/screen_large_corpus_v10.py') == sha(study / 'source.py')
    assert protocol['parser_source_sha256'] == sha(root / 'training/generate_large_corpus_v10.py')
    assert protocol['generation_report_sha256'] == sha(ev / 'authored-response-pilot-66/report.json')
    assert protocol['generation_protocol_sha256'] == sha(ev / 'authored-response-pilot-66/protocol.json')
    for key in ('training_ready', 'independent_annotation', 'stable_release_ready'):
        assert protocol[key] is False
    rows = records(source)
    raw = records(rawdir / 'judgments.jsonl')
    screened = records(rawdir / 'screened-families.jsonl')
    assert len(rows) == len(screened) == 16 and len(raw) == 32
    queries = {r['key']: r for r in preflight['queries']}
    by_key = {r['key']: r for r in raw}
    assert len(queries) == len(by_key) == 32
    expected_keys = {r['request']['family'] + ':' + language for r in rows for language in ('ro', 'uk')}
    assert set(by_key) == set(queries) == expected_keys
    outcomes = []
    accepted_groups = 0
    counts = Counter()
    for row, saved in zip(rows, screened, strict=True):
        request = row['request']
        assert row['accepted'] is True and request['partition'] == 'train'
        assert request['task'] == 'response_behavior'
        pair_pass = []
        for language in ('ro', 'uk'):
            key = request['family'] + ':' + language
            record = by_key[key]
            assert record['input_sha256'] == digest(row)
            assert record['query_token_sha256'] == queries[key]['query_token_sha256']
            expected = [request['category'], 'appropriate']
            if int(digest({'family': request['family'], 'language': language})[-1], 16) % 2:
                expected.reverse()
            try:
                actual = parse_raw(record['raw'])
            except (ValueError, TypeError):
                actual = None
            assert record['judgment'] == actual
            accepted = actual is not None and actual['labels'] == expected and all(actual['natural'])
            assert type(record['accepted']) is bool and record['accepted'] is accepted
            assert (record['error'] is None) == (actual is not None)
            pair_pass.append(accepted)
            outcomes.append({'key': key, 'expected_labels': expected, 'judgment': actual,
                             'accepted': accepted, 'malformed': actual is None})
        passed = all(pair_pass)
        assert saved == {**row, 'machine_screen_pass': passed,
                         'semantic_labels_verified': False, 'independent_annotation': False}
        accepted_groups += passed
        counts['response_behavior:train:' + str(passed)] += 1
    assert report['screened_groups'] == 16 and report['accepted_groups'] == accepted_groups
    assert report['counts'] == dict(counts)
    assert report['screened_sha256'] == sha(rawdir / 'screened-families.jsonl')
    assert report['judgments_sha256'] == sha(rawdir / 'judgments.jsonl')
    return {'verification': 'PASS', 'scope': 'Saved source, bindings, query hashes, raw judgment parsing and all recorded machine-screen outcomes; no independent semantic annotation or model rerun.',
            'verifier_sha256': sha(__file__), 'report_sha256': sha(study / 'report.json'),
            'protocol_sha256': sha(study / 'protocol.json'), 'judgments_sha256': report['judgments_sha256'],
            'screened_sha256': report['screened_sha256'], 'language_pairs': 32,
            'accepted_language_pairs': sum(r['accepted'] for r in outcomes),
            'accepted_groups': accepted_groups, 'malformed_judgments': sum(r['malformed'] for r in outcomes),
            'outcomes': outcomes, 'training_ready': False, 'semantic_labels_verified': False,
            'independent_annotation': False, 'stable_release_ready': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite verification')
    result = verify(args.root, args.private)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'outcomes'}))


if __name__ == '__main__':
    main()
