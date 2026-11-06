"""Evaluate a selected head on hash-bound exposed frozen-encoder features.

No model fitting or threshold selection occurs here. Reuse saves encoder compute;
it neither creates an independent test nor measures end-to-end inference latency.
"""
import argparse
import json
from pathlib import Path

from roguard.gemma4_review import _score
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_evaluation import evaluate, paired_compare
from training.evaluate_semantic_readout_v10 import world_from_logits
from training.shared_tail_training_v9 import make_head


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cached_study', 'cached_private', 'original', 'selected', 'selected_study', 'study'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    p.add_argument('--consensus-study', type=Path)
    a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite evaluation')
    cached = json.loads((a.cached_study/'report.json').read_text())
    selection = json.loads((a.selected_study/'report.json').read_text())
    if not cached['complete'] or not selection['complete']:
        raise ValueError('Extraction and selection must be complete')
    consensus = None
    if a.consensus_study is not None:
        fit = json.loads((a.consensus_study/'report.json').read_text())
        if not fit['complete'] or file_sha(a.consensus_study/'consensus.json') != fit['consensus_sha256']:
            raise ValueError('Consensus selection is incomplete or changed')
        consensus = json.loads((a.consensus_study/'consensus.json').read_text())
        if consensus['format'] != 'roguard-semantic-consensus-v10-research1':
            raise ValueError('Unknown consensus format')
    original_protocol = json.loads((a.cached_study/'protocol.json').read_text())
    if original_protocol.get('format') != 'roguard-v10-semantic-regression-1':
        raise ValueError('Only the verified frozen-v9 encoder cache supports this evaluation')
    paths = {'broad': Path('research/v10-evidence/broad-v9-regression/cases.jsonl'),
        **{name: Path('research/v10-evidence/data')/(name+'.jsonl')
           for name in ('dev', 'challenge', 'v9-regression')}}
    for name, path in paths.items():
        if file_sha(path) != original_protocol['inputs'][name]:
            raise ValueError('Evaluation population binding differs')
    for lang in ('ro', 'uk'):
        for suffix, sha in cached['features'][lang].items():
            if file_sha(a.cached_private/(lang+'-'+suffix)) != sha:
                raise ValueError('Cached feature binding differs')
    a.study.mkdir(parents=True)
    protocol = {'format': 'roguard-semantic-cached-evaluation-1',
        'source_sha256': file_sha(Path(__file__)),
        'extraction_report_sha256': file_sha(a.cached_study/'report.json'),
        'selection_report_sha256': file_sha(a.selected_study/'report.json'),
        'input_hashes': original_protocol['inputs'], 'fitting_performed': False,
        'threshold_selection_performed': False, 'end_to_end_latency_measured': False,
        'independent_test': False, 'stable_release_ready': False}
    if consensus is not None:
        protocol['consensus_sha256'] = file_sha(a.consensus_study/'consensus.json')
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    import mlx.core as mx
    from safetensors.numpy import load_file
    baseline = {name: read_jsonl(a.cached_study/(name+'-baseline.jsonl')) for name in paths}
    previous = {name: {r['checkpoint_id']: r for r in rows} for name, rows in baseline.items()}
    predictions = {name: [] for name in paths}
    parent_predictions = {name: [] for name in paths}
    for lang in ('ro', 'uk'):
        original = load_file(a.original/'actual'/(lang+'-selected.safetensors'))
        selected = load_file(a.selected/(lang+'-selected.safetensors'))
        if set(original) != set(selected) or any(not np.array_equal(original[k], selected[k])
                for k in original if not k.startswith('heads.'+lang+'.')):
            raise ValueError('Cannot reuse features after an encoder change')
        spec = json.loads((a.selected_study/(lang+'-review.json')).read_text())
        if spec['adapter_sha256'] != file_sha(a.selected/(lang+'-selected.safetensors')):
            raise ValueError('Selected head binding differs')
        if consensus is not None and (consensus['languages'][lang]['adapter_sha256'] != spec['adapter_sha256']
                or consensus['languages'][lang]['review_spec_sha256'] != file_sha(a.selected_study/(lang+'-review.json'))):
            raise ValueError('Consensus belongs to another selected model')
        reference_spec = json.loads((a.cached_study/'v9-binding'/(lang+'-review.json')).read_text())
        if reference_spec['adapter_sha256'] != file_sha(a.original/'actual'/(lang+'-selected.safetensors')):
            raise ValueError('Cached encoder has another original adapter binding')
        head = make_head({k: selected['heads.'+lang+'.'+v] for k, v in
            {'0.weight': 'first.weight', '0.bias': 'first.bias',
             '3.weight': 'last.weight', '3.bias': 'last.bias'}.items()})
        head.eval()
        x = np.load(a.cached_private/(lang+'-features.npy'), mmap_mode='r')
        old_logits = np.load(a.cached_private/(lang+'-original-logits.npy'), mmap_mode='r')
        mappings = read_jsonl(a.cached_private/(lang+'-rows.jsonl'))
        if len(mappings) != len(x) or len(old_logits) != len(x):
            raise ValueError('Missing feature mappings')
        # Match installed single-prefix inference, including GEMM batch shape.
        for start in range(len(x)):
            values = head(mx.array(x[start:start+1])); mx.eval(values)
            for row, logits in zip(mappings[start:start+1], np.asarray(values)):
                old = previous[row['population']][row['checkpoint_id']]
                if old['input_digest'] != row['input_digest'] or old['inference_input_digest'] != row['inference_input_digest']:
                    raise ValueError('Cached prediction input binding differs')
                old_score = _score(old_logits[start], reference_spec)
                if abs(old['score']-old_score)>1e-7 or old['predicted'] != int(old_score>=reference_spec['threshold']):
                    raise ValueError('Cached baseline score differs from its frozen logits')
                score = _score(logits, spec)
                parent = {**old, 'score': score,
                    'predicted': int(score >= spec['threshold']), 'model_identity': spec['adapter_sha256'],
                    'interpretation_set': {'hypotheses': [world_from_logits(logits)],
                        'reason': 'UNCALIBRATED_ARGMAX_DIAGNOSTIC',
                        'coverage_assumptions_verified': False, 'outside_ontology_ruled_out': False}}
                parent_predictions[row['population']].append(parent)
                candidate = dict(parent)
                if consensus is not None:
                    from training.semantic_consensus_v10 import score_features, predict
                    parameters = consensus['languages'][lang]
                    score = float(predict(score_features(logits[None, :], spec['couplings']), parameters)[0])
                    candidate.update(score=score, predicted=int(score >= parameters['threshold']),
                        consensus_identity=protocol['consensus_sha256'])
                predictions[row['population']].append(candidate)
    report = {'complete': True, 'populations': {}, 'stable_release_ready': False,
              'fitting_performed': False, 'independent_test': False}
    for name, path in paths.items():
        cases = read_jsonl(path)
        write_jsonl(a.study/(name+'-candidate.jsonl'), predictions[name])
        report['populations'][name] = {'baseline': evaluate(cases, baseline[name]),
            'candidate': evaluate(cases, predictions[name]),
            'comparison': paired_compare(cases, baseline[name], predictions[name])}
        if consensus is not None:
            report['populations'][name].update(parent=evaluate(cases, parent_predictions[name]),
                parent_comparison=paired_compare(cases, parent_predictions[name], predictions[name]))
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name: {label: {lang: r['strict_accuracy'] for lang, r in result[label]['groups'].items()}
          for label in ('baseline', 'candidate')} for name, result in report['populations'].items()}, indent=2))


if __name__ == '__main__':
    main()
