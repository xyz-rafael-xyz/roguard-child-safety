"""Audit semantic support, cached-head TRAIN fit and exposed DEV wording descriptors."""
import argparse
from collections import Counter
import json
import gzip
from pathlib import Path
import re
import time
import unicodedata


def descriptor(payload):
    text = unicodedata.normalize('NFC', payload['turns'][-1]['text']).casefold()
    positions = [text.find(mark) for mark in ('«', '„', '“', '"') if mark in text]
    prefix = text[:min(positions)] if positions else text
    return {'has_opening_quote': bool(positions),
            'negation_before_first_quote': bool(re.search(r'\b(?:nu|не)\b', prefix))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--private', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite audit')
    from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
    from roguard.evidence import ONTOLOGY
    from roguard.gemma4_review import _score
    from roguard.assurance import digest
    plan = json.loads(a.plan.read_text())
    for base, bindings in ((a.root, plan['public']), (a.private, plan['private'])):
        for name, expected in bindings.items():
            if file_sha(base/name) != expected: raise ValueError('Bound source changed: '+name)
    import numpy as np
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from safetensors.numpy import load_file
    from training.shared_tail_training_v9 import make_head
    fields = tuple(ONTOLOGY); sizes = tuple(map(len, ONTOLOGY.values()))
    labels = {lang: read_jsonl(a.private/'crossed-review-features-62'/(lang+'-rows.jsonl')) for lang in ('ro', 'uk')}
    historical = {}
    for name in ('shared-tail-corpus-164', 'grounding-tail-corpus-174', 'referent-tail-corpus-176', 'reported-tail-corpus-195'):
        with gzip.open(a.root/'research/v9-stabilization'/name/'inputs.jsonl.gz', 'rt') as stream:
            for line in stream:
                row = json.loads(line)['row']; key = (row['language'], row['case_id'])
                if key in historical: raise ValueError('Duplicate historical source identity')
                historical[key] = (row['source'], digest(row))
    descriptors = {}
    with (a.private/'large-review-input-recovery-72/actual/inputs.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line); key = (row['language'], row['feature_index'])
            if key in descriptors: raise ValueError('Duplicate recovered position')
            reference = labels[key[0]][key[1]]
            if any(reference[k] != row[k] for k in reference): raise ValueError('Recovered labels or position changed')
            source = historical.get((row['language'], row['case_id']))
            if source is not None and source[1] != row['source_row_sha256']: raise ValueError('Historical source row changed')
            descriptors[key] = {**descriptor(row['payload']), 'source_name': row['source_name'],
                                'source_kind': source[0] if source is not None else row['group']}
    if len(descriptors) != sum(map(len, labels.values())): raise ValueError('Incomplete recovered inputs')
    a.output.mkdir(parents=True); started = time.monotonic(); records = []; languages = {}
    expected_fit = json.loads((a.root/'research/v10-evidence/dropout-readout-75/report.json').read_text())
    for lang in ('ro', 'uk'):
        metadata = labels[lang]; train = [i for i, r in enumerate(metadata) if r['partition'] == 'train']
        dev = [i for i, r in enumerate(metadata) if r['partition'] == 'dev']
        x = np.load(a.private/'crossed-review-features-62'/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        teacher = np.load(a.private/'crossed-review-features-62'/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        selected = load_file(a.private/'dropout-readout-75'/(lang+'-selected.safetensors'))
        mapping = {'0.weight': 'first.weight', '0.bias': 'first.bias', '3.weight': 'last.weight', '3.bias': 'last.bias'}
        head = make_head({key: selected['heads.'+lang+'.'+value] for key, value in mapping.items()}); head.eval()
        spec = json.loads((a.root/'research/v10-evidence/dropout-readout-75'/(lang+'-review.json')).read_text())
        old_spec = json.loads((a.root/'research/v10-evidence/crossed-review-features-62/v9-binding'/(lang+'-review.json')).read_text())
        support = Counter(tuple(metadata[i]['fields']) for i in train)
        result = {'train_worlds': len(support), 'dev_worlds': len({tuple(metadata[i]['fields']) for i in dev}),
                  'all_ontology_worlds': int(np.prod(sizes)),
                  'train_world_counts': [{'fields': list(world), 'rows': count} for world, count in sorted(support.items())],
                  'partitions': {}}
        languages[lang] = result
        for split, indices in (('train', train), ('dev', dev)):
            if split == 'dev':
                z = np.load(a.private/'dropout-readout-75'/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
            else:
                parts = []
                for start in range(0, len(indices), 256):
                    output = head(mx.array(x[indices[start:start+256]])); mx.eval(output); parts.append(np.asarray(output).copy())
                z = np.concatenate(parts).astype(np.float64)
            if z.shape != (len(indices), sum(sizes)+2) or not np.isfinite(z).all(): raise ValueError('Invalid logits')
            np.save(a.output/(lang+'-'+split+'-logits.npy'), z, allow_pickle=False)
            pred = np.column_stack([part.argmax(axis=1) for part in np.split(z[:, :sum(sizes)], np.cumsum(sizes)[:-1], axis=1)])
            gold = np.array([metadata[i]['fields'] for i in indices]); wrong = pred != gold
            descriptor_counts = Counter(); confusions = [Counter() for _ in fields]; binary_errors = baseline_errors = 0
            unseen_rows = unseen_errors = 0; formal_collapse = 0; binary_with_correct_fields = 0
            for j, index in enumerate(indices):
                row = metadata[index]; desc = descriptors[(lang, index)]
                score = _score(z[j], spec); old_score = _score(teacher[index], old_spec)
                decision, baseline = int(score >= spec['threshold']), int(old_score >= old_spec['threshold'])
                binary_errors += decision != row['gold']; baseline_errors += baseline != row['gold']
                binary_with_correct_fields += decision != row['gold'] and not bool(wrong[j].any())
                unseen = tuple(row['fields']) not in support
                unseen_rows += unseen; unseen_errors += unseen and bool(wrong[j].any())
                collapse = row['fields'][2] == 2 and int(pred[j, 2]) == 0
                formal_collapse += row['group'] == 'formal' and collapse
                # This is a literal wording descriptor, not a scope parser or semantic label.
                key = (desc['source_kind'], row['group'], ONTOLOGY['assertion'][row['fields'][2]],
                       desc['has_opening_quote'], desc['negation_before_first_quote'])
                descriptor_counts[(*key, 'rows')] += 1
                descriptor_counts[(*key, 'assertion_errors')] += bool(wrong[j, 2])
                for k in range(len(fields)): confusions[k][(int(gold[j, k]), int(pred[j, k]))] += 1
                records.append({'language': lang, 'case_id': row['case_id'], 'feature_index': index,
                                'partition': split, 'group': row['group'], 'token_sha256': row['token_sha256'],
                                'gold': row['gold'], 'fields': row['fields'], 'predicted_fields': pred[j].tolist(),
                                'candidate_score': score, 'candidate': decision, 'v9': baseline,
                                'v9_score': old_score, 'train_world_support': support.get(tuple(row['fields']), 0), **desc})
            groups = sorted({key[:-1] for key in descriptor_counts})
            summary = {'rows': len(indices), 'binary_errors': int(binary_errors), 'v9_binary_errors': int(baseline_errors),
                       'joint_field_errors': int(wrong.any(axis=1).sum()), 'field_errors': wrong.sum(axis=0).tolist(),
                       'binary_errors_with_all_fields_correct': int(binary_with_correct_fields),
                       'rows_with_unseen_worlds': int(unseen_rows), 'joint_errors_with_unseen_worlds': int(unseen_errors),
                       'formal_uncertain_to_affirmed_errors': int(formal_collapse),
                       'field_confusions': {field: [{'gold': ONTOLOGY[field][g], 'predicted': ONTOLOGY[field][v], 'rows': n}
                                                     for (g, v), n in sorted(confusions[k].items())] for k, field in enumerate(fields)},
                       'wording_groups': [{'source_kind': source_kind, 'group': group, 'assertion_gold': assertion, 'has_opening_quote': quoted,
                                           'negation_before_first_quote': negation,
                                           'rows': descriptor_counts[(source_kind, group, assertion, quoted, negation, 'rows')],
                                           'assertion_errors': descriptor_counts[(source_kind, group, assertion, quoted, negation, 'assertion_errors')]}
                                          for source_kind, group, assertion, quoted, negation in groups],
                       'logits_sha256': file_sha(a.output/(lang+'-'+split+'-logits.npy'))}
            result['partitions'][split] = summary
            if split == 'dev':
                selected_record = expected_fit['languages'][lang]['selected']
                if (summary['binary_errors'] != selected_record['binary_errors']
                        or summary['joint_field_errors'] != selected_record['joint_argmax_errors']
                        or summary['field_errors'] != selected_record['field_errors']): raise ValueError('Saved DEV recount changed')
    write_jsonl(a.output/'predictions.jsonl', records)
    report = {'complete': True, 'source_sha256': file_sha(__file__), 'plan_sha256': file_sha(a.plan),
              'languages': languages, 'predictions_sha256': file_sha(a.output/'predictions.jsonl'),
              'seconds': time.monotonic()-started, 'device': 'cpu', 'encoder_loaded': False,
              'training_performed': False, 'train_head_inference_performed': True, 'dev_logits_reused': True,
              'held_out_test_opened': False, 'qualified': False,
              'scope': 'Exact semantic-label support and cached-head TRAIN fit with exposed saved DEV recount. '
                       'Negation-before-first-quote is a literal descriptor, not a semantic annotation. '
                       'No new gold validation, candidate fitting, threshold changes or causal explanation is established.'}
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({lang: {split: {k: v for k, v in pop.items() if k in ('rows', 'binary_errors', 'joint_field_errors', 'formal_uncertain_to_affirmed_errors')}
                             for split, pop in r['partitions'].items()} for lang, r in languages.items()}, indent=2))


if __name__ == '__main__': main()
