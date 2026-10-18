"""Audit held-out input overlap without scoring models or displaying messages."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from roguard.context import render
from roguard.semantic import normalize_semantic_text

PROBES = {
    'contrast': ('research/v8-stabilization/contrast-probe.jsonl',
                 '7bf47bd773d5aceed684ad0aaeced6b85dc7d1602acfe17312a47073fe77e42f'),
    'families_original': ('research/v8-stabilization/context-families-01/test.jsonl',
                         '3786bfdc01a0421668df04eae8263488d4cb744bffb54192d647daf48d4fa40d'),
    'families_lexical_v2': ('research/v8-stabilization/context-families-01/test-lexical-v2.jsonl',
                           '290cbb6dd7ccdccdf9c4371e4569b49831fbcb8eff5822d1dba5104819669b38'),
}


def normalize(text):
    return normalize_semantic_text(text).translate(str.maketrans('şţŞŢ', 'șțȘȚ'))


def keys(row):
    focus = row.get('focus', len(row['turns'])-1)
    turns = row['turns'][:focus+1]
    language = row['language']
    def key(value):
        return hashlib.sha256(json.dumps([language, value], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return key([normalize(render(turns)), normalize(turns[-1]['text'])]), key(normalize(turns[-1]['text'])), {
        key(normalize(t['text'])) for t in turns}


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def audit(training_paths):
    probes = {}
    for name, (filename, expected) in PROBES.items():
        path = Path(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Reserved input changed: '+name)
        probes[name] = [(r['language'], keys(r)) for r in read(path)]
    results = {}
    for path in training_paths:
        contexts, focals, all_turns, languages = set(), set(), set(), set()
        count = 0
        for row in read(path):
            context, focal, turns = keys(row)
            contexts.add(context); focals.add(focal); all_turns.update(turns)
            languages.add(row['language']); count += 1
        if not count:
            raise ValueError('Empty training history')
        results[str(path)] = dict(rows=count, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            languages=sorted(languages), probes={})
        for name, rows in probes.items():
            relevant = [k for language, k in rows if language in languages]
            results[str(path)]['probes'][name] = dict(probe_rows_in_language_scope=len(relevant),
                exact_normalized_input_overlap=sum(k[0] in contexts for k in relevant),
                focal_overlap=sum(k[1] in focals for k in relevant),
                focal_seen_in_any_training_turn=sum(k[1] in all_turns for k in relevant))
    cross = {}
    for left, rows in probes.items():
        a = {k[0] for _, k in rows}
        for right, others in probes.items():
            if left < right:
                cross[left+' / '+right] = len(a & {k[0] for _, k in others})
    return dict(recorded_utc=datetime.now(timezone.utc).isoformat(), training_histories=results,
        reserved_input_hashes={name: digest for name, (_, digest) in PROBES.items()},
        exact_input_overlap_between_probe_sets=cross,
        raw_datasets_read_programmatically=True, message_text_displayed_or_saved=False,
        reserved_predictions_produced=False, training_data_or_selection_changed=False,
        scope='Exact normalized input and focal overlap only; not semantic/family independence or independent annotation.',
        analyst_blind=False, source_visibility_record='research/v9-stabilization/reserved-probe-access-audit.json',
        stable_release_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-history', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.training_history)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
