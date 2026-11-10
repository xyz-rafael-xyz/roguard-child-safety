"""Recover bound TRAIN/DEV payloads and exact tokens; no encoder or fitting."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return [json.loads(line) for line in raw.decode().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'private', 'base', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Refuse overwrite'
    from roguard.assurance import digest
    from roguard.gemma4_contract import classifier_tokens, BASE_REVISION, FILES
    from training.native_fields_v9 import targets
    from training.prepare_instruction_sft_v8 import target
    from transformers import AutoTokenizer
    evidence = args.root / 'research/v10-evidence'
    plan_path = args.private / 'large-review-input-recovery-72/plan.json'
    plan = json.loads(plan_path.read_text())
    original_protocol_path = evidence / 'semantic-features-11/protocol.json'
    feature_report_path = evidence / 'crossed-review-features-62/report.json'
    assert sha(original_protocol_path) == plan['original_protocol_sha256']
    assert sha(feature_report_path) == plan['feature_report_sha256']
    assert sha(args.root / 'src/roguard/gemma4_contract.py') == plan['runtime_tokenizer_source_sha256']
    original_protocol = json.loads(original_protocol_path.read_text())
    feature_report = json.loads(feature_report_path.read_text())
    assert feature_report['complete'] and args.base.name == BASE_REVISION
    tokenizer_files = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        tokenizer_files[name] = sha(args.base / name)
        assert tokenizer_files[name] == FILES[name]['sha256']
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    sources, inventory = {}, {}

    def add(row, partition, group, source, old_token=None, old_count=None):
        assert partition in ('train', 'dev')
        assert row['language'] in ('ro', 'uk')
        key = (row['language'], row['case_id'])
        assert key not in inventory, ('Duplicate source identity', key)
        assert type(row['focus']) is int and 0 <= row['focus'] < len(row['turns'])
        inventory[key] = {'row': row, 'partition': partition, 'group': group, 'source': source,
                          'token_sha256': old_token, 'token_count': old_count}

    for relative, expected in original_protocol['cache_inputs'].items():
        path = args.root / relative
        assert sha(path) == expected
        entries = read(path)
        sources[relative] = {'sha256': expected, 'rows': len(entries)}
        for entry in entries:
            add(entry['row'], entry['partition'], entry['group'], relative,
                entry['token_sha256'], entry['token_count'])
    for name in ('compositional-review-56', 'crossed-review-61'):
        report_path = evidence / name / 'report.json'
        data = json.loads(report_path.read_text())
        assert data['complete'] and data['format'] == 'roguard-compositional-review-1'
        sources[name + '/report.json'] = {'sha256': sha(report_path)}
        for partition in ('train', 'dev'):
            path = args.private / name / (partition + '.jsonl')
            assert sha(path) == data['artifacts'][partition]
            rows = read(path)
            source_name = name + '/' + partition + '.jsonl'
            sources[source_name] = {'sha256': sha(path), 'rows': len(rows)}
            for row in rows:
                assert row['partition'] == partition and row['review'] == target(row)
                add(row, partition, row['source'], source_name)
    assert len(inventory) == plan['rows'] == 37766
    metadata = []
    for language in ('ro', 'uk'):
        path = args.private / 'crossed-review-features-62' / (language + '-rows.jsonl')
        assert sha(path) == feature_report['languages'][language]['artifacts']['rows.jsonl']
        rows = read(path)
        assert len(rows) == feature_report['languages'][language]['rows']
        metadata.extend((index, row) for index, row in enumerate(rows))
    assert len(metadata) == len(inventory)
    assert {(r['language'], r['case_id']) for _, r in metadata} == set(inventory)
    assert shutil.disk_usage(args.private).free > 8 * 1024**3
    args.output.mkdir()
    started = time.monotonic()
    lengths, counts, hashes = [], Counter(), set()
    preserved_nonfinal_focus = 0
    destination = args.output / 'inputs.jsonl'
    try:
        with destination.open('x') as stream:
            for count, (index, meta) in enumerate(metadata, 1):
                entry = inventory[(meta['language'], meta['case_id'])]
                row = entry['row']
                assert meta['partition'] == entry['partition'] and meta['group'] == entry['group']
                assert meta['gold'] == target(row) and meta['fields'] == targets([row])[0]
                turns = row['turns'][:row['focus'] + 1]
                payload = {'language': row['language'], 'turns': turns}
                tokens = classifier_tokens(tokenizer, payload)
                key = digest(tokens)
                assert key == meta['token_sha256'], ('Token identity differs', meta['case_id'])
                if entry['token_sha256'] is not None:
                    assert key == entry['token_sha256'] and len(tokens) == entry['token_count']
                assert 0 < len(tokens) <= 1024
                assert (row['language'], key) not in hashes
                hashes.add((row['language'], key))
                preserved_nonfinal_focus += row['focus'] != len(row['turns']) - 1
                output = {**meta, 'feature_index': index, 'payload': payload, 'tokens': tokens,
                          'token_count': len(tokens), 'source_name': entry['source'],
                          'source_row_sha256': digest(row), 'source_focus': row['focus']}
                stream.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + '\n')
                lengths.append(len(tokens))
                counts[row['language'] + ':' + meta['partition']] += 1
                if count % 1000 == 0:
                    assert shutil.disk_usage(args.private).free > 8 * 1024**3
                    print(json.dumps({'done': count, 'total': len(metadata), 'seconds': time.monotonic() - started}), flush=True)
        assert sum(v for k, v in counts.items() if k.endswith(':train')) == plan['training_rows']
        assert sum(v for k, v in counts.items() if k.endswith(':dev')) == plan['development_rows']
        report = {'complete': True, 'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(plan_path),
                  'feature_report_sha256': sha(feature_report_path), 'original_protocol_sha256': sha(original_protocol_path),
                  'sources': sources, 'tokenizer_files': tokenizer_files, 'base_revision': BASE_REVISION,
                  'rows': len(metadata), 'groups': dict(counts), 'inputs_sha256': sha(destination),
                  'inputs_bytes': destination.stat().st_size, 'unique_language_token_pairs': len(hashes),
                  'token_count': {'minimum': min(lengths), 'maximum': max(lengths)},
                  'nonfinal_focus_prefixes_preserved': preserved_nonfinal_focus,
                  'every_token_hash_matches_feature_store': True, 'seconds': time.monotonic() - started,
                  'scope': 'Exact source/label/partition/token recovery only; no new examples or independent label validation.',
                  'encoder_loaded': False, 'training_experiment_registered': False, 'training_launched': False,
                  'held_out_test_opened': False, 'stable_release_ready': False}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    except Exception as error:
        (args.output / 'failure.json').write_text(json.dumps({'complete': False, 'error_type': type(error).__name__,
            'error': str(error), 'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(plan_path)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
