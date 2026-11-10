"""Verify matched curriculum labels and screen the actual inference views."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from roguard.trajectory_data import file_sha, prefix, read_jsonl, write_jsonl
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory
from training.prepare_semantic_augmentation_v10 import key
from training.prepare_instruction_sft_v8 import target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--selector', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite an input audit')
    report = json.loads((a.training/'report.json').read_text())
    if file_sha(a.training/'train.jsonl') != report['training_data_sha256']:
        raise ValueError('Authored training corpus differs')
    rows = read_jsonl(a.training/'train.jsonl'); families = defaultdict(list)
    for row in rows:
        if row['partition'] != 'train' or target(row) != row['review']:
            raise ValueError('Training partition or policy label differs')
        families[row['family']].append(row)
    for family in families.values():
        if len(family) != 4 or {r['attributes']['assertion'] for r in family} != {'AFFIRMED', 'NEGATED', 'UNCERTAIN', 'RETRACTED'}:
            raise ValueError('Incomplete matched assertion family')
        bodies, attributes, neutral_contexts = set(), set(), set()
        for row in family:
            text = row['turns'][row['focus']]['text']; body = text[text.index('«')+1:text.rindex('»')]
            bodies.add(body)
            attributes.add(tuple(row['attributes'][f] for f in ('subject', 'source', 'time', 'concern')))
            if row['attributes']['assertion'] == 'RETRACTED':
                if row['attributes']['relation'] != 'RETRACTS' or '«'+body+'»' not in row['turns'][0]['text']:
                    raise ValueError('Retraction does not reference its prior assertion')
            else:
                if row['attributes']['relation'] != 'NEW':
                    raise ValueError('Non-retraction relation differs')
                neutral_contexts.add(json.dumps(row['turns'][:row['focus']], sort_keys=True))
        if len(bodies) != 1 or len(attributes) != 1 or len(neutral_contexts) != 1:
            raise ValueError('Matched family content or context differs')
    selector = TrajectoryModel(a.selector); protected = set(); old_train = set(); inputs = {}
    for name in ('shared-tail-corpus-164', 'grounding-tail-corpus-174', 'referent-tail-corpus-176', 'reported-tail-corpus-195'):
        path = Path('research/v9-stabilization')/name/'inputs.jsonl.gz'; inputs[str(path)] = file_sha(path)
        for entry in map(json.loads, gzip.open(path, 'rt', encoding='utf-8')):
            row = entry['row']; payload = {'language': row['language'], 'turns': row['turns'][:row['focus']+1]}
            (old_train if entry['partition'] == 'train' else protected).add(key(payload))
    earlier = Path('research/v10-evidence/semantic-training-12/train.jsonl'); inputs[str(earlier)] = file_sha(earlier)
    old_train.update(key(row) for row in read_jsonl(earlier))
    paths = [Path('research/v10-evidence/data')/(name+'.jsonl') for name in ('dev', 'calibration', 'challenge', 'v9-regression')]
    paths.append(Path('research/v10-evidence/broad-v9-regression/cases.jsonl'))
    for path in paths:
        inputs[str(path)] = file_sha(path)
        for case in read_jsonl(path):
            for point in case['checkpoints']:
                payload = prefix(case, point['focus']); view, inference = prepare_memory(payload, selector, 1200, preserve_complete=True)
                if view['mandatory_overflow']:
                    raise ValueError('Unsupported protected input')
                protected.update((key(payload), key(inference)))
    accepted, exclusions, seen = [], [], {}; changed = 0
    for row in rows:
        payload = {'language': row['language'], 'turns': row['turns'][:row['focus']+1]}
        view, inference = prepare_memory(payload, selector, 1200, preserve_complete=True)
        if view['mandatory_overflow'] or not inference['turns'] or inference['turns'][-1] != payload['turns'][-1]:
            exclusions.append({'case_id': row['case_id'], 'reason': 'focal_text_not_preserved'}); continue
        identity = key(inference)
        if identity in protected or identity in old_train:
            exclusions.append({'case_id': row['case_id'], 'reason': 'protected_or_prior_training_inference'}); continue
        if identity in seen:
            if seen[identity] != row['attributes']:
                raise ValueError('Conflicting labels for an identical inference view')
            exclusions.append({'case_id': row['case_id'], 'reason': 'duplicate_new_inference'}); continue
        seen[identity] = row['attributes']; changed += inference != payload
        accepted.append({**row, **inference, 'focus': len(inference['turns'])-1,
                         'original_prefix_key': key(payload), 'inference_prefix_key': identity})
    a.output.mkdir(parents=True); write_jsonl(a.output/'train.jsonl', accepted)
    result = {'format': 'roguard-scope-input-audit-1', 'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-18.json'),
        'authored_training_report_sha256': file_sha(a.training/'report.json'),
        'training_data_sha256': file_sha(a.output/'train.jsonl'), 'selector_sha256': selector.identity,
        'protected_input_sources': inputs, 'authored_rows': len(rows), 'accepted_rows': len(accepted),
        'matched_families_checked': len(families), 'changed_memory_views': changed,
        'exclusions': exclusions, 'language_counts': dict(Counter(r['language'] for r in accepted)),
        'exact_protected_inference_overlap': 0, 'independent_language_review': False,
        'new_held_out_claim': False, 'stable_release_ready': False}
    (a.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != 'protected_input_sources'}, indent=2))


if __name__ == '__main__':
    main()
