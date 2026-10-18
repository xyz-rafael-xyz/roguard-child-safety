"""Replay fusion over paired actual installed neural outputs, without loading models."""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .review_fusion_v9 import features, probability
    from .reserved_metrics_v9 import counts
else:
    from review_fusion_v9 import features, probability
    from reserved_metrics_v9 import counts


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--installed-root',type=Path,required=True)
    p.add_argument('--language',choices=('ro','uk'),required=True)
    a = p.parse_args()
    import roguard.review_fusion as runtime
    if a.installed_root.resolve() not in Path(runtime.__file__).resolve().parents:
        raise ValueError('Fusion runtime was not imported from the installed wheel')
    root = a.root; study = root/'review-fusion-38'; output = study/a.language
    frozen = json.loads((study/'source-freeze.json').read_text())
    if sha(runtime.__file__) != frozen['src/roguard/review_fusion.py']:
        raise ValueError('Installed fusion source differs from the frozen runtime')
    exported = json.loads((output/'export.json').read_text()); artifact = output/'fusion.json'
    model = runtime.ReviewFusion(artifact,expected_artifact_sha256=exported['artifact_sha256'])
    spec = json.loads(artifact.read_text())
    native_dir = root/'installed-native-29'/a.language
    interpreter_dir = root/'installed-interpreter-30/screen'
    nr = json.loads((native_dir/'report.json').read_text()); ir = json.loads((interpreter_dir/'report.json').read_text())
    if (nr.get('parity_pass') is not True or ir.get('parity_pass') is not True
            or nr['artifact_sha256'] != spec['native_artifact_sha256']
            or ir['archive_sha256'] != spec['interpreter_archive_sha256']):
        raise ValueError('Installed component parity or identity differs')
    data = root.parent/'v8-stabilization/instruction-sft-01/screen.jsonl'
    if sha(data) != ir['data_sha256']:
        raise ValueError('Screen inputs differ from installed interpreter replay')
    rows = [r for r in read(data) if r['language'] == a.language]
    def indexed(paths):
        values = [row for path in paths for row in read(path)]
        result = {row['case_id']:row for row in values}
        if len(result) != len(values):raise ValueError('Duplicate prediction identity')
        return result
    native = indexed([native_dir/(split+'-predictions.jsonl') for split in ('dev','cal')])
    interpreter = indexed([interpreter_dir/'predictions.jsonl'])
    reference_native = indexed([root/('native-full-22-'+a.language)/(split+'-predictions.jsonl') for split in ('dev','cal')])
    selection = json.loads((root/'interpreter-full-21/selection.json').read_text())
    parent = 'interpreter-14' if selection['selected_study'] == 14 else 'interpreter-policy-27'
    reference_interpreter = indexed([root/parent/'adapted/predictions.jsonl'])
    if sha(root/parent/'adapted/predictions.jsonl') != ir['reference_sha256']:
        raise ValueError('Interpreter reference changed')
    for split in ('dev','cal'):
        if sha(root/('native-full-22-'+a.language)/(split+'-predictions.jsonl')) != nr['splits'][split]['reference_sha256']:
            raise ValueError('Native reference changed')
    results, changes, errors = [], 0, []
    for row in rows:
        identity = row['case_id']; n,s = native[identity],interpreter[identity]
        rn,rs = reference_native[identity],reference_interpreter[identity]
        if not n['gold'] == s['gold'] == rn['gold'] == rs['gold']:
            raise ValueError('Neural output gold labels differ')
        if (n['score'] is None or abs(n['score']-rn['score']) > 1e-4 or n['predicted'] != rn['predicted']
                or s['interpretation'] != rs['interpretation'] or s['valid_label_tokens'] != rs['valid_label_tokens']
                or s['predicted'] != rs['predicted']):
            raise ValueError('Upstream component parity changed')
        for field, values in s['conditional_field_probabilities'].items():
            if max(abs(x-y) for x,y in zip(values,rs['conditional_field_probabilities'][field])) > .001:
                raise ValueError('Upstream interpreter probability parity changed')
        value = model.predict(dict(language=a.language,turns=row['turns']),n,s,focus=row['focus'])
        reference_score = probability(spec['model'],features(rn,rs))
        reference_label = None if reference_score is None else int(reference_score >= spec['threshold'])
        changes += value['predicted'] != reference_label
        if reference_score is None or value['score'] is None:
            if reference_score is not value['score']:raise ValueError('Abstention parity differs')
        else:errors.append(abs(reference_score-value['score']))
        results.append(dict(case_id=identity,gold=n['gold'],kind='conversational' if row.get('source')=='v7_training_frame_projection' else 'formal',**value))
    if not results:raise ValueError('Empty fusion screen')
    error = max(errors,default=0.0)
    report = dict(format='roguard-installed-fusion-replay-v1',rows=len(results),language=a.language,
        decision_changes=changes,maximum_score_error=error,score_tolerance=.001,parity_pass=changes==0 and error<=.001,
        screen_counts={kind:counts([r for r in results if r['kind']==kind]) for kind in ('formal','conversational')},
        artifact_sha256=sha(artifact),native_runtime_report_sha256=sha(native_dir/'report.json'),
        interpreter_runtime_report_sha256=sha(interpreter_dir/'report.json'),installed_runtime_sha256=sha(runtime.__file__),
        actual_predictions_sha256={'native_'+split:sha(native_dir/(split+'-predictions.jsonl')) for split in ('dev','cal')},
        interpreter_predictions_sha256=sha(interpreter_dir/'predictions.jsonl'),data_sha256=sha(data),
        wheel_sha256=ir['wheel_sha256'],actual_installed_neural_outputs_used=True,neural_models_loaded_in_this_check=False,
        independent_accuracy_evaluation=False,reserved_probes_used=False,stable_release_ready=False)
    with (output/'screen-predictions.jsonl').open('x') as stream:
        stream.write(''.join(json.dumps(r)+'\n' for r in results))
    (output/'runtime-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['parity_pass']:raise SystemExit('Fusion runtime parity failed')


if __name__ == '__main__':
    main()
