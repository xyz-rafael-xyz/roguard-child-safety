"""Prove when selection retained the published model despite new serialization."""
import argparse
import json
from pathlib import Path

from roguard.trajectory_data import file_sha, read_jsonl
from roguard.trajectory_evaluation import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('selected_study','selected','original','reference','challenge','study'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists():raise ValueError('Refusing to overwrite identity assessment')
    import numpy as np
    from safetensors.numpy import load_file
    selection=json.loads((a.selected_study/'report.json').read_text())
    reference=json.loads((a.reference/'report.json').read_text())
    protocol=json.loads((a.reference/'protocol.json').read_text())
    challenge=json.loads((a.challenge/'report.json').read_text())
    if not all(r['complete'] for r in (selection,reference,challenge)):
        raise ValueError('Parent study incomplete')
    if file_sha(a.challenge/'predictions.jsonl')!=challenge['predictions_sha256']:
        raise ValueError('Fresh challenge binding differs')
    languages={}
    for lang in ('ro','uk'):
        source=a.original/'actual'/(lang+'-selected.safetensors');candidate=a.selected/(lang+'-selected.safetensors')
        original_spec=json.loads((a.reference/'v9-binding'/(lang+'-review.json')).read_text())
        selected_spec=json.loads((a.selected_study/(lang+'-review.json')).read_text())
        if (selection['languages'][lang]['selected']['epoch']!=0
                or file_sha(source)!=original_spec['adapter_sha256']
                or file_sha(candidate)!=selected_spec['adapter_sha256']
                or selected_spec['adapter_sha256']!=selection['languages'][lang]['selected']['adapter_sha256']):
            raise ValueError('Expected bound epoch-zero selection')
        if {k:v for k,v in original_spec.items() if k!='adapter_sha256'}!={k:v for k,v in selected_spec.items() if k!='adapter_sha256'}:
            raise ValueError('Runtime configuration changed')
        old=load_file(source);new=load_file(candidate)
        if len(old)!=80 or set(old)!=set(new) or any(
                old[k].shape!=new[k].shape or old[k].dtype!=new[k].dtype or old[k].dtype!=np.float32
                or not np.isfinite(new[k]).all() or old[k].tobytes()!=new[k].tobytes() for k in old):
            raise ValueError('A selected tensor differs from the published model')
        languages[lang]={'original_file_sha256':file_sha(source),'selected_file_sha256':file_sha(candidate),
            'tensors_checked':len(old),'all_tensor_bytes_exact':True,'runtime_config_equal_except_file_hash':True}
    records=read_jsonl(a.challenge/'predictions.jsonl')
    if len(records)!=72 or any(r['candidate']!=r['v9'] for r in records):
        raise ValueError('Fresh selected-component outputs differ from v9')
    populations={};input_hashes={};prediction_hashes={}
    for name in ('broad','dev','challenge','v9-regression'):
        cases_path=(Path('research/v10-evidence/broad-v9-regression/cases.jsonl') if name=='broad'
                    else Path('research/v10-evidence/data')/(name+'.jsonl'))
        if file_sha(cases_path)!=protocol['inputs'][name]:raise ValueError('Reference population differs')
        pred_path=a.reference/(name+'-baseline.jsonl')
        # Recompute descriptive scores; no new inference or changed-candidate claim.
        populations[name]=evaluate(read_jsonl(cases_path),read_jsonl(pred_path))
        input_hashes[name]=file_sha(cases_path);prediction_hashes[name]=file_sha(pred_path)
    a.study.mkdir(parents=True)
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    report={'complete':True,'source_sha256':file_sha(__file__),
        'selection_report_sha256':file_sha(a.selected_study/'report.json'),
        'reference_report_sha256':file_sha(a.reference/'report.json'),
        'challenge_report_sha256':file_sha(a.challenge/'report.json'),
        'languages':languages,'fresh_challenge_outputs_equal_v9':72,'fresh_challenge_errors':challenge['totals']['candidate_errors'],
        'reused_reference_inputs':input_hashes,'reused_reference_predictions':prediction_hashes,
        'populations':populations,'new_neural_regression_required':False,
        'reason':'Both selected epoch-zero models have identical tensor bytes and runtime settings to published v9; only serialization hashes differ. No selected model improvement occurred.',
        'decision':'REJECT','independent_validation':False,'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('populations','reused_reference_inputs','reused_reference_predictions')},indent=2))


if __name__=='__main__':main()
