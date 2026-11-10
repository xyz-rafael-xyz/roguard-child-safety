"""Compose training-only surfaces, screening every protected prefix for overlap."""
import argparse
from collections import Counter
import gzip
import itertools
import json
from pathlib import Path
import random
import re
import unicodedata

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, prefix, read_jsonl, write_jsonl
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory
from training.augment_evidence_v8 import PHRASES
from training.evidence_v8 import CONCERNS, TEXT, SUBJECTS, TIMES, ASSERTIONS
from training.prepare_instruction_sft_v8 import target


def key(payload):
    return digest({'language': payload['language'], 'turns': [
        {'role': t['role'], 'text': re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', t['text']).casefold()).strip()}
        for t in payload['turns']]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selector', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a training dataset')
    selector = TrajectoryModel(args.selector)
    roots = [Path('research/v9-stabilization')/name for name in
             ('shared-tail-corpus-164', 'grounding-tail-corpus-174',
              'referent-tail-corpus-176', 'reported-tail-corpus-195')]
    old_train, protected, inputs = set(), set(), {}
    for root in roots:
        path = root/'inputs.jsonl.gz'; inputs[str(path)] = file_sha(path)
        for entry in map(json.loads, gzip.open(path, 'rt', encoding='utf-8')):
            row = entry['row']
            payload = {'language': row['language'], 'turns': row['turns'][:row['focus']+1]}
            (old_train if entry['partition'] == 'train' else protected).add(key(payload))
    exposure = {}
    paths = [Path('research/v10-evidence/data')/(name+'.jsonl') for name in
             ('dev', 'calibration', 'challenge', 'v9-regression')]
    paths.append(Path('research/v10-evidence/broad-v9-regression/cases.jsonl'))
    for path in paths:
        inputs[str(path)] = file_sha(path)
        count = 0; overlap = []
        for row in read_jsonl(path):
            for point in row['checkpoints']:
                payload = prefix(row, point['focus'])
                view, inference = prepare_memory(payload, selector, 1200, preserve_complete=True)
                if view['mandatory_overflow']:
                    raise ValueError('Unsupported protected input')
                count += 1
                if key(inference) in old_train:
                    overlap.append(row['case_id']+':'+str(point['focus']))
                protected.update((key(payload), key(inference)))
        exposure[str(path)] = {'checkpoints': count, 'old_finetuning_train_overlap': overlap}
    # All content and all surface phrases below come from previously declared
    # TRAIN banks. No dev/cal/test surface is imported from WORDS/SPLITS.
    rng = random.Random(1012); candidates = []
    for lang in ('ro', 'uk'):
        for s, t, a, signal, variant in itertools.product(range(3), range(3), range(4), range(2), range(6)):
            body = CONCERNS[lang][rng.randrange(12)] if signal else TEXT[lang][rng.randrange(24)]
            subject = PHRASES[lang]['subject'][s][variant]
            temporal = PHRASES[lang]['time'][t][(variant+t)%6]
            assertion = PHRASES[lang]['assertion'][a][(variant+s)%6]
            parts = [subject, temporal, assertion+' «'+body+'»']; rng.shuffle(parts)
            prior = (subject+' '+temporal+' '+PHRASES[lang]['assertion'][0][variant]+' «'+body+'»'
                     if a == 3 else TEXT[lang][rng.randrange(24)])
            attributes = {'subject': SUBJECTS[s], 'time': TIMES[t], 'assertion': ASSERTIONS[a],
                'source': ('DIRECT', 'REPORTED', 'FICTION')[s],
                'relation': 'RETRACTS' if a == 3 else 'NEW', 'concern': str(signal)}
            row = {'case_id': 'semantic12-'+lang+'-'+str(len(candidates)),
                'language': lang, 'turns': [
                    {'role': 'minor', 'text': prior},
                    {'role': 'assistant', 'text': 'Te ascult.' if lang == 'ro' else 'Я слухаю.'},
                    {'role': 'minor', 'text': ' '.join(parts)}],
                'focus': 2, 'attributes': attributes, 'partition': 'train',
                'source': 'training_bank_composition_v10', 'synthetic_only': True,
                'independently_annotated': False}
            row['review'] = target(row); candidates.append(row)
    training_path = Path('research/v10-evidence/data/train.jsonl')
    inputs[str(training_path)] = file_sha(training_path)
    for case in read_jsonl(training_path):
        for point in case['checkpoints']:
            if point.get('world') is None:
                continue
            view, inference = prepare_memory(prefix(case, point['focus']), selector, 1200, preserve_complete=True)
            if view['mandatory_overflow']:
                continue
            row = {'case_id': 'memory12-'+case['case_id']+':'+str(point['focus']),
                   **inference, 'focus': len(inference['turns'])-1,
                   'attributes': point['world'], 'review': point['review'], 'partition': 'train',
                   'source': 'existing_v10_training_only_memory', 'synthetic_only': True,
                   'independently_annotated': False}
            if target(row) != point['review']:
                raise ValueError('Existing training label conflicts with ontology')
            candidates.append(row)
    seen = set(old_train); rows = []; excluded = Counter()
    for row in candidates:
        identity = key(row)
        if identity in protected:
            excluded['protected_prefix'] += 1
        elif identity in seen:
            excluded['existing_or_duplicate_training_prefix'] += 1
        else:
            seen.add(identity); rows.append(row)
    args.output.mkdir(parents=True)
    write_jsonl(args.output/'train.jsonl', rows)
    report = {'format': 'roguard-semantic-augmentation-v10-1', 'seed': 1012,
        'source_sha256': file_sha(Path(__file__)), 'inputs': inputs,
        'training_data_sha256': file_sha(args.output/'train.jsonl'),
        'training_rows': len(rows), 'counts': dict(Counter(r['language']+':'+r['source'] for r in rows)),
        'excluded': dict(excluded), 'old_training_unique_prefixes': len(old_train),
        'protected_unique_prefixes': len(protected), 'exposed_evaluation_overlap_audit': exposure,
        'independent_language_review': False, 'new_held_out_claim': False,
        'limitation': 'Exact normalized prefix exclusion does not establish family or author independence.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs', 'exposed_evaluation_overlap_audit')}, indent=2))


if __name__ == '__main__':
    main()
