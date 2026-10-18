"""Replay an installed field mixer over bound actual installed neural outputs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

if __package__:
    from .reserved_metrics_v9 import counts
else:
    from reserved_metrics_v9 import counts


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','installed-root','wheel','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--language',choices=('ro','uk'),required=True);args=parser.parse_args()
    import roguard.field_fusion as runtime
    import roguard.native_fields as neural_runtime
    if args.installed_root.resolve() not in Path(runtime.__file__).resolve().parents:
        raise ValueError('Field mixer must come from isolated installed wheel')
    freeze=json.loads((args.root/'joint-field-fusion-57/source-freeze.json').read_text())
    if sha(runtime.__file__)!=freeze['src/roguard/field_fusion.py']:
        raise ValueError('Installed field mixer differs from frozen source')
    study=args.root/'joint-field-fusion-57'/args.language
    selection=json.loads((study/'selection.json').read_text());audit=json.loads((study/'audit.json').read_text())
    artifact=study/'fusion.json';spec=json.loads(artifact.read_text())
    if selection['artifact_sha256']!=sha(artifact) or audit['selection_sha256']!=sha(study/'selection.json'):
        raise ValueError('Selected field mixture changed')
    parents=[args.root/name/args.language for name in ('installed-native-fields-45','installed-joint-fields-55')]
    reports=[json.loads((parent/'report.json').read_text()) for parent in parents]
    for report,key in zip(reports,('native_heads_sha256','pretrained_heads_sha256')):
        if (report.get('complete') is not True or report['artifact_sha256']!=spec[key]
                or report['runtime_source_sha256']!=sha(neural_runtime.__file__) or set(report['splits'])!={'dev','cal'}):
            raise ValueError('Installed component identity differs')
        for value in report['splits'].values():
            # Binary parent decisions are not predictive features of this mixer.
            if (value['abstentions'] or not 0<=value['maximum_field_error']<=1e-4
                    or not 0<=value['maximum_score_error']<=1e-4):
                raise ValueError('Installed component field probabilities differ')
    model=runtime.FieldReviewFusion(artifact,expected_sha256=sha(artifact))
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(format='roguard-installed-field-fusion-replay-v1',language=args.language,
        artifact_sha256=sha(artifact),selection_sha256=sha(study/'selection.json'),
        wheel_sha256=sha(args.wheel),runtime_source_sha256=sha(runtime.__file__),
        installed_module=str(Path(runtime.__file__).resolve()),
        component_report_sha256=[sha(parent/'report.json') for parent in parents],
        cached_actual_installed_neural_outputs_used=True,fresh_neural_inference=False,
        synthetic_only=True,stable_release_ready=False,reserved_probes_used=False,splits={})
    for split in ('dev','cal'):
        data=args.root/('native-full-22-'+args.language)/(split+'.jsonl.gz')
        expected_data=selection['data_sha256']['dev'] if split=='dev' else audit['calibration_data_sha256']
        if sha(data)!=expected_data:raise ValueError('Evaluation inputs changed')
        rows=[json.loads(line) for line in gzip.decompress(data.read_bytes()).splitlines()]
        reference_path=study/(split+'-predictions.jsonl');reference=read(reference_path)
        components=[]
        for parent,report in zip(parents,reports):
            path=parent/(split+'-predictions.jsonl')
            if sha(path)!=report['splits'][split]['predictions_sha256']:raise ValueError('Neural output bytes changed')
            components.append(read(path))
        if any([r['case_id'] for r in part]!=[r['case_id'] for r in rows] for part in [reference,*components]):
            raise ValueError('Neural input coverage differs')
        outputs=[]
        for row,ref,left,right in zip(rows,reference,*components):
            if ref['gold']!=left['gold'] or ref['gold']!=right['gold']:raise ValueError('Component labels differ')
            value=model.predict(dict(language=args.language,turns=row['turns']),left,right,focus=row['focus'])
            outputs.append(dict(case_id=row['case_id'],gold=ref['gold'],kind=ref['kind'],**value))
        changes=sum(v['predicted']!=r['predicted'] for v,r in zip(outputs,reference))
        abstentions=sum(v['predicted'] is None for v in outputs)
        error=max(abs(v['score']-r['score']) for v,r in zip(outputs,reference))
        field_error=max(abs(x-y) for v,r in zip(outputs,reference) for field in v['field_probabilities']
                        for x,y in zip(v['field_probabilities'][field],r['field_probabilities'][field]))
        groups={kind:counts([v for v in outputs if v['kind']==kind]) for kind in ('formal','conversational')}
        path=args.output/(split+'-predictions.jsonl');path.write_text(''.join(json.dumps(v)+'\n' for v in outputs))
        result['splits'][split]=dict(decision_changes=changes,abstentions=abstentions,
            maximum_score_error=error,maximum_field_error=field_error,counts=groups,
            parity_pass=not changes and not abstentions and error<=1e-4 and field_error<=1e-4,
            accuracy_gates_pass=all(v['passes'] for v in groups.values()),predictions_sha256=sha(path),
            input_sha256=sha(data),reference_sha256=sha(reference_path))
    result.update(complete=True,parity_pass=all(v['parity_pass'] for v in result['splits'].values()),
                  accuracy_gates_pass=all(v['accuracy_gates_pass'] for v in result['splits'].values()))
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['parity_pass']:raise SystemExit('Installed field-fusion parity failed')


if __name__=='__main__':main()
