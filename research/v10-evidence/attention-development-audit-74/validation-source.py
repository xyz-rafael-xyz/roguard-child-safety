"""Validate the new attention audit on already retained study71 DEV outputs."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite verification')
    import numpy as np
    from eval.audit_attention_development_v10 import recount
    from roguard.gemma4_review import _score
    from roguard.trajectory_data import read_jsonl
    plan = json.loads(a.plan.read_text())
    for name, value in plan['sources'].items():
        if sha(a.root/name) != value: raise ValueError('Source changed: '+name)
    e = a.root/'research/v10-evidence'
    study = e/'policy-consistent-readout-71'
    reference = json.loads((study/'report.json').read_text())
    results = {}; maximum_error = 0.
    for lang in ('ro', 'uk'):
        bound = plan['inputs'][lang]
        for name, value in bound['public'].items():
            if sha(a.root/name) != value: raise ValueError('Public input changed')
        for name, value in bound['private'].items():
            if sha(a.private/name) != value: raise ValueError('Private input changed')
        rows = read_jsonl(a.private/'crossed-review-features-62'/(lang+'-rows.jsonl'))
        indices = [i for i, r in enumerate(rows) if r['partition'] == 'dev']; dev = [rows[i] for i in indices]
        teacher = np.load(a.private/'crossed-review-features-62'/(lang+'-logits.npy'), allow_pickle=False)
        spec = json.loads((e/'crossed-review-features-62/v9-binding'/(lang+'-review.json')).read_text())
        baseline = [int(_score(teacher[i], spec) >= spec['threshold']) for i in indices]
        logits = np.load(a.private/'policy-consistent-readout-71'/(lang+'-selected-dev-logits.npy'), allow_pickle=False)
        summary, predictions, scores, error = recount(dev, logits, spec['couplings'], baseline)
        selected = reference['languages'][lang]['selected']
        keys = ('binary_errors', 'joint_argmax_errors', 'field_errors', 'mean_field_cross_entropy',
                'mode', 'threshold', 'score_margin', 'selection_rank')
        if any(summary[k] != selected[k] for k in keys): raise ValueError('Independent selected statistics differ')
        expected = [{'case_id': row['case_id'], 'language': lang, 'group': row['group'], 'gold': row['gold'],
                     'predicted': int(pred), 'score': float(score)} for row, pred, score in zip(dev, predictions, scores)]
        if expected != read_jsonl(study/(lang+'-development-predictions.jsonl')):
            raise ValueError('Independent per-case scores or decisions differ')
        if summary['fixed_v9'] != plan['expected'][lang]['fixed_v9'] or summary['introduced_v9'] != plan['expected'][lang]['introduced_v9']:
            raise ValueError('Independent paired counts differ')
        results[lang] = {'rows': len(dev), 'recount': summary, 'maximum_policy_enumeration_error': error,
                         'all_saved_statistics_and_prediction_records_exact': True}
        maximum_error = max(maximum_error, error)
    report = {'complete': True, 'verified': True, 'plan_sha256': sha(a.plan), 'source_sha256': sha(__file__),
              'languages': results, 'rows_verified': sum(r['rows'] for r in results.values()),
              'maximum_policy_enumeration_error': maximum_error,
              'independent_threshold_implementation': True, 'encoder_loaded': False,
              'new_model_inference': False, 'training_performed': False, 'held_out_test_opened': False,
              'completed73_audit_performed': False, 'qualified': False,
              'scope': 'Recount of exposed saved71 DEV logits, thresholds, selection statistics, policy semantics and paired decisions. '
                       'No new accuracy population, independent gold labels or completed73 result.'}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'verified': True, 'rows': report['rows_verified'], 'maximum_policy_enumeration_error': maximum_error}))


if __name__ == '__main__': main()
