"""Freeze a qualifying fusion before any reserved neural predictions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_counts(groups):
    if set(groups) != {'formal', 'conversational'}:
        raise ValueError('Missing required fusion strata')
    for group in groups.values():
        for key in ('positive', 'negative', 'missed', 'false_reviews', 'abstentions'):
            if type(group.get(key)) is not int or group[key] < 0:
                raise ValueError('Invalid fusion counts')
        p, n = group['positive'], group['negative']
        if (not p or not n or group['missed'] > p or group['false_reviews'] > n
                or group['abstentions'] > group['missed']+group['false_reviews']
                or group['missed']*100 > p or group['false_reviews']*20 > n
                or group.get('passes') is not True):
            raise ValueError('Fusion accuracy gates have not passed')


def qualify(root):
    """Check accuracy and actual installed component/composite identity together."""
    root = Path(root)
    evidence = {}
    def read(relative):
        path = root/relative
        evidence[relative] = sha(path)
        return json.loads(path.read_text())
    ir = read('installed-interpreter-30/screen/report.json')
    selected = read('interpreter-full-21/selection.json')
    if (ir.get('parity_pass') is not True or ir.get('archive_sha256') != selected['archive_sha256']
            or ir.get('decision_changes') != 0 or ir.get('schema_changes') != 0
            or ir.get('interpretation_changes') != 0 or ir.get('unexpected_context_abstentions') != 0
            or not 0 <= ir.get('maximum_probability_error', float('inf')) <= .001):
        raise ValueError('Installed interpreter did not reproduce selected model')
    sources = {'src/roguard/interpreter.py': ir['runtime_source_sha256']}
    candidates = {}
    for language in ('ro', 'uk'):
        selection_path = 'review-fusion-36/'+language+'/selection.json'
        selection = read(selection_path)
        audit = read('review-fusion-36/'+language+'/audit.json')
        require_counts(selection['selected']['metrics'])
        require_counts(audit['calibration'])
        if audit.get('counts_pass') is not True or audit['selection_sha256'] != evidence[selection_path]:
            raise ValueError('Fusion calibration is not bound to selected fit')
        artifact_path = 'review-fusion-38/'+language+'/fusion.json'
        spec = read(artifact_path)
        fr = read('review-fusion-38/'+language+'/runtime-report.json')
        nr_path = 'installed-native-29/'+language+'/report.json'
        nr = read(nr_path)
        native_artifact = 'native-full-22-'+language+'/review.json.gz'
        evidence[native_artifact] = sha(root/native_artifact)
        if (fr.get('parity_pass') is not True or fr.get('decision_changes') != 0
                or not 0 <= fr.get('maximum_score_error', float('inf')) <= .001
                or fr.get('actual_installed_neural_outputs_used') is not True
                or fr['artifact_sha256'] != evidence[artifact_path]
                or fr['native_runtime_report_sha256'] != evidence[nr_path]
                or fr['interpreter_runtime_report_sha256'] != evidence['installed-interpreter-30/screen/report.json']
                or fr['wheel_sha256'] != ir['wheel_sha256']
                or spec['selection_sha256'] != evidence[selection_path]
                or spec['model'] != {k:selection['model'][k] for k in ('mean','scale','coefficients')}
                or spec['threshold'] != selection['selected']['threshold']
                or spec['native_artifact_sha256'] != evidence[native_artifact]
                or spec['interpreter_archive_sha256'] != selected['archive_sha256']
                or nr.get('complete') is not True or nr.get('parity_pass') is not True
                or nr['artifact_sha256'] != evidence[native_artifact]
                or set(nr['splits']) != {'dev', 'cal'}):
            raise ValueError('Installed composite identity or parity differs')
        for split in nr['splits'].values():
            if (split.get('parity_pass') is not True or split['abstentions']
                    or split['decision_disagreements'] or not 0 <= split['max_score_error'] <= 1e-4):
                raise ValueError('Installed native parity differs')
        # Individual parent accuracy need not pass when the fitted ensemble does.
        for name, value in [('src/roguard/native_review.py', nr['runtime_source_sha256']),
                            ('src/roguard/review_fusion.py', fr['installed_runtime_sha256'])]:
            if name in sources and sources[name] != value:
                raise ValueError('Installed language runtimes differ')
            sources[name] = value
        candidates[language] = dict(artifact=artifact_path, artifact_sha256=evidence[artifact_path],
            native_artifact=native_artifact, native_artifact_sha256=evidence[native_artifact],
            checkpoint='native-full-22-'+language+'/selected-model.safetensors')
    number = selected['selected_study']
    if number not in (14, 27):
        raise ValueError('Unexpected selected interpreter')
    archive = ('interpreter-14' if number == 14 else 'interpreter-policy-27')+'/selected-adapter.json.gz'
    if sha(root/archive) != selected['archive_sha256']:
        raise ValueError('Interpreter archive changed')
    return dict(format='roguard-reserved-fusion-plan-v1', recorded_utc=datetime.now(timezone.utc).isoformat(),
        candidates=candidates, interpreter_archive=archive, interpreter_archive_sha256=selected['archive_sha256'],
        evidence_sha256=evidence, runtime_source_sha256=sources,
        verifier_source_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ('reserved_fusion_v9.py','evaluate_reserved_fusion_v9.py','reserved_metrics_v9.py','verify_reserved_native_v9.py')},
        synthetic_only=True, analyst_blind=False, stable_release_ready=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=qualify(a.root)
    with a.output.open('x') as stream:stream.write(json.dumps(plan,indent=2)+'\n')
    print('Frozen reserved fusion plan SHA256',sha(a.output))


if __name__=='__main__':main()
