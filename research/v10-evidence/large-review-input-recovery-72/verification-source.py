"""Independently inspect the serialized recovered payload/token inventory."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    value = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return [json.loads(line) for line in value.decode().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refuse overwrite'
    from roguard.assurance import digest
    from roguard.gemma4_contract import PREFIX_TOKENS
    folder = a.private / 'large-review-input-recovery-72'
    report = json.loads((folder / 'actual/report.json').read_text())
    assert report['complete'] and report['source_sha256'] == sha(a.private / 'recover-large-review-inputs-72.py')
    assert report['plan_sha256'] == sha(folder / 'plan.json')
    assert report['inputs_sha256'] == sha(folder / 'actual/inputs.jsonl')
    assert report['encoder_loaded'] is report['training_launched'] is report['held_out_test_opened'] is False
    evidence = a.root / 'research/v10-evidence'
    feature_report_path = evidence / 'crossed-review-features-62/report.json'
    assert sha(feature_report_path) == report['feature_report_sha256']
    feature_report = json.loads(feature_report_path.read_text())
    metadata = {}
    for lang in ('ro', 'uk'):
        path = a.private / 'crossed-review-features-62' / (lang + '-rows.jsonl')
        assert sha(path) == feature_report['languages'][lang]['artifacts']['rows.jsonl']
        metadata[lang] = read(path)
    original = {}
    source_names = {}
    for source, info in report['sources'].items():
        if source.startswith('research/v9-stabilization/'):
            path = a.root / source
            assert sha(path) == info['sha256']
            entries = read(path)
            assert len(entries) == info['rows']
            rows = [r['row'] for r in entries]
        elif source.endswith('/report.json'):
            assert sha(evidence / source) == info['sha256']
            continue
        else:
            path = a.private / source
            assert sha(path) == info['sha256']
            rows = read(path)
            assert len(rows) == info['rows']
        for row in rows:
            key = (row['language'], row['case_id'])
            assert key not in original
            original[key] = row
            source_names[key] = source
    seen, token_keys, groups = set(), set(), Counter()
    minimum, maximum, common_prefix = 10**9, 0, 0
    per_language_index = {'ro': 0, 'uk': 0}
    with (folder / 'actual/inputs.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['language'], row['case_id'])
            assert key not in seen
            seen.add(key)
            index = per_language_index[row['language']]
            assert row['feature_index'] == index
            expected = metadata[row['language']][index]
            assert all(row[k] == v for k, v in expected.items())
            source = original[key]
            assert row['source_name'] == source_names[key]
            assert row['source_row_sha256'] == digest(source)
            assert row['source_focus'] == source['focus']
            assert row['payload'] == {'language': source['language'], 'turns': source['turns'][:source['focus'] + 1]}
            tokens = row['tokens']
            assert len(tokens) == row['token_count'] and all(type(t) is int and t >= 0 for t in tokens)
            assert digest(tokens) == expected['token_sha256']
            assert 0 < len(tokens) <= 1024
            token_key = (row['language'], row['token_sha256'])
            assert token_key not in token_keys
            token_keys.add(token_key)
            groups[row['language'] + ':' + row['partition']] += 1
            per_language_index[row['language']] += 1
            minimum, maximum = min(minimum, len(tokens)), max(maximum, len(tokens))
            common_prefix += tokens[:len(PREFIX_TOKENS)] == PREFIX_TOKENS
    assert seen == set(original)
    assert dict(groups) == report['groups']
    assert len(seen) == report['rows'] == report['unique_language_token_pairs'] == 37766
    assert per_language_index == {lang: len(rows) for lang, rows in metadata.items()}
    assert report['token_count'] == {'minimum': minimum, 'maximum': maximum}
    proof = {'complete': True, 'verified': True, 'source_sha256': sha(Path(__file__)),
             'report_sha256': sha(folder / 'actual/report.json'), 'inputs_sha256': report['inputs_sha256'],
             'rows_verified': len(seen), 'groups': dict(groups),
             'all_serialized_payloads_match_source_focus_prefixes': True,
             'all_serialized_tokens_match_frozen_feature_hashes': True,
             'all_feature_indices_partitions_labels_and_groups_match': True,
             'runtime_common_prefix_matching_rows': common_prefix,
             'runtime_common_prefix_tokens': len(PREFIX_TOKENS),
             'scope': 'Stored-file hash, every source payload/focus, metadata index/label/partition and token identity. No tokenization or model inference is repeated by this verifier.',
             'encoder_loaded': False, 'training_experiment_registered': False,
             'training_launched': False, 'held_out_test_opened': False, 'stable_release_ready': False}
    with a.output.open('x') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
