"""Independently recompute the fixed unanimity rule from bound parent records."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from roguard.gemma4_review import _score
from roguard.trajectory_data import file_sha, read_jsonl
from roguard.trajectory_evaluation import checkpoint_id
from training.prepare_instruction_sft_v8 import target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'features', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); root = Path('research/v10-evidence')
    if a.output.exists():
        raise ValueError('Refusing to overwrite unanimity assessment')
    protocol = json.loads((a.study/'protocol.json').read_text())
    report = json.loads((a.study/'report.json').read_text())
    if not report['complete'] or report['selected'] != 'unanimous-three-models' or protocol['rules'] != ['unanimous-three-models']:
        raise ValueError('Expected the single frozen unanimity hypothesis')
    if file_sha(a.study/'source.py') != protocol['source_sha256'] or file_sha(a.study/'selection.json') != report['selection_sha256']:
        raise ValueError('Study source or frozen selection changed')
    for mapping in ('inputs', 'parent_reports', 'component_specs'):
        for path, expected in protocol[mapping].items():
            actual_path = Path(path)/'report.json' if mapping == 'parent_reports' else Path(path)
            if file_sha(actual_path) != expected:
                raise ValueError('Bound study input changed: '+path)
    feature_study = root/'base-reference-features-18b'
    feature_report = json.loads((feature_study/'report.json').read_text())
    import numpy as np
    teacher = {}; labels = {}; reference_specs = {}; readout = {}; attention = {}
    for lang in ('ro', 'uk'):
        for suffix, expected in feature_report['languages'][lang]['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix)) != expected:
                raise ValueError('Original DEV feature binding changed')
        spec = json.loads((feature_study/'v9-binding'/(lang+'-review.json')).read_text()); reference_specs[lang] = spec
        rows = read_jsonl(a.features/(lang+'-rows.jsonl')); z = np.load(a.features/(lang+'-logits.npy'), allow_pickle=False)
        for row, values in zip(rows, z):
            if row['partition'] == 'dev':
                key = (lang, row['case_id']); labels[key] = row['gold']
                teacher[key] = int(_score(values, spec) >= spec['threshold'])
        for name, destination in (('semantic-readout-11', readout), ('semantic-tail-14', attention)):
            component_spec = json.loads((root/name/(lang+'-review.json')).read_text())
            for row in read_jsonl(root/name/(lang+'-development-predictions.jsonl')):
                key = (lang, row['case_id'])
                if row['gold'] != labels[key] or row['predicted'] != int(row['score'] >= component_spec['threshold']):
                    raise ValueError('Development component score/label changed')
                destination[key] = row['predicted']
    original_gold = {}
    for name in ('shared-tail-corpus-164', 'grounding-tail-corpus-174', 'referent-tail-corpus-176', 'reported-tail-corpus-195'):
        for entry in map(json.loads, gzip.open(Path('research/v9-stabilization')/name/'inputs.jsonl.gz', 'rt', encoding='utf-8')):
            if entry['partition'] == 'dev':
                row = entry['row']; original_gold[(row['language'], row['case_id'])] = target(row)
    if len(original_gold) != 6341 or any(labels[k] != v for k, v in original_gold.items()):
        raise ValueError('Original development gold differs from source annotations')
    dev_parent = {r['checkpoint_id']: r for r in read_jsonl(root/'semantic-regression-11/dev-candidate.jsonl')}
    for key in set(labels)-set(original_gold):
        readout[key] = dev_parent[key[1]]['predicted']
    if set(teacher) != set(readout) or set(teacher) != set(attention) or len(teacher) != 6668:
        raise ValueError('Eligible development coverage differs')
    result = {'format': 'roguard-unanimity-assessment-1', 'complete': True,
        'source_sha256': file_sha(Path(__file__)), 'study_report_sha256': file_sha(a.study/'report.json'),
        'study_protocol_sha256': file_sha(a.study/'protocol.json'), 'populations': {},
        'fitting_performed': False, 'fresh_neural_inference_performed': False,
        'independent_test': False, 'stable_release_ready': False}
    def check(name, records, gold, parents, expected):
        observed = {(r['language'], r['case_id']): r for r in records}
        if len(observed) != len(records) or set(observed) != set(gold):
            raise ValueError('Incomplete or duplicated population: '+name)
        errors = []; fixed = []; introduced = []; groups = Counter()
        for key, row in observed.items():
            if row['gold'] != gold[key] or any(row[k] != parents[k][key] for k in parents):
                raise ValueError('Recorded vote or gold differs from the parent source')
            # Separate arithmetic implementation; do not import the fitted-study rule.
            value = 1 if sum(parents[k][key] for k in ('v9', 'readout', 'attention')) == 3 else 0
            if 'predicted' in row and row['predicted'] != value:
                raise ValueError('Saved unanimity output differs')
            if value != gold[key]:
                errors.append(key[1])
            if value == gold[key] and parents['v9'][key] != gold[key]:
                fixed.append(key[1])
            if value != gold[key] and parents['v9'][key] == gold[key]:
                introduced.append(key[1])
            groups[key[0]] += 1
        if (len(records) != expected['rows'] or len(errors) != expected['errors']
                or len(fixed) != expected['fixed'] or len(introduced) != expected['introduced']
                or any(groups[l] != expected['groups'][l]['total'] for l in groups)):
            raise ValueError('Stored summary differs from recomputed predictions')
        result['populations'][name] = {'rows': len(records), 'errors': errors, 'fixed': fixed, 'introduced': introduced}
    check('eligible-development', read_jsonl(a.study/'development-components.jsonl'), labels,
          {'v9': teacher, 'readout': readout, 'attention': attention}, report['development']['unanimous-three-models'])
    attention_report = json.loads((root/'semantic-tail-regression-14/report.json').read_text())
    for name in ('broad', 'challenge', 'v9-regression'):
        path = root/'broad-v9-regression/cases.jsonl' if name == 'broad' else root/'data'/(name+'.jsonl')
        gold = {(c['language'], checkpoint_id(c, point)): point['review'] for c in read_jsonl(path) for point in c['checkpoints']}
        parents = {}
        for kind, study, suffix in (('v9', 'semantic-regression-11', 'baseline'),
                                    ('readout', 'semantic-regression-11', 'candidate'),
                                    ('attention', 'semantic-tail-regression-14', 'candidate')):
            parent_path = root/study/(name+'-'+suffix+'.jsonl'); values = read_jsonl(parent_path)
            if kind == 'attention':
                if file_sha(parent_path) != attention_report['prediction_sha256'][name]:
                    raise ValueError('Attention prediction hash differs')
                for row in values:
                    spec = json.loads((root/'semantic-tail-14'/(row['language']+'-review.json')).read_text())
                    value = _score(row['field_logits'], spec)
                    if abs(value-row['score']) > 1e-7 or int(value >= spec['threshold']) != row['predicted']:
                        raise ValueError('Attention saved logits and score differ')
            parents[kind] = {(r['language'], r['checkpoint_id']): r['predicted'] for r in values}
        check(name, read_jsonl(a.study/(name+'-predictions.jsonl')), gold, parents, report['regression'][name])
    result['decision'] = 'BINARY_ENGINEERING_PASS_ONLY' if all(not p['errors'] and not p['introduced'] for p in result['populations'].values()) else 'REJECT'
    result['remaining_requirements'] = ['FRESH_INSTALLED_RUNTIME_PARITY', 'SEMANTIC_SET_QUALIFICATION',
        'RESPONSE_QUALIFICATION', 'FRESH_CALIBRATION', 'INDEPENDENT_FINAL_EVALUATION', 'MEASURED_REVIEWER_BENEFIT']
    a.output.mkdir(parents=True)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
