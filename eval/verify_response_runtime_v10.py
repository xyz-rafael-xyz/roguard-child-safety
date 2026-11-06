"""Compare an installed response runtime against frozen training features/logits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from roguard.assurance import digest
from roguard.gemma4_response import Gemma4ResponseModel
import roguard.gemma4_response as runtime
from roguard.gemma4_review import Gemma4ReviewModel
from roguard.trajectory_data import file_sha,read_jsonl,write_jsonl


def payload(row):
    return {'language':row['language'],'turns':([{'role':'minor','text':row['context']}] if row['context'] else [])+
            [{'role':'assistant','text':row['draft']}]}


def select_probes(rows,identities):
    if len(rows)!=len(identities):raise ValueError('Missing reference input mapping')
    groups=defaultdict(list)
    for i,(row,identity) in enumerate(zip(rows,identities)):
        if identity!={'case_id':row['case_id'],'partition':row['partition'],'input_sha256':digest(payload(row))}:
            raise ValueError('Reference row identity differs')
        if row['partition'] not in ('train','dev'):raise ValueError('Only exposed TRAIN/DEV probes are allowed')
        groups[(row.get('origin','historical'),row['partition'],row['language'],row['gold'])].append(i)
    if not groups:raise ValueError('Empty probe population')
    return sorted({indices[0] for indices in groups.values()}|{indices[-1] for indices in groups.values()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact','weights','reference_artifact','reference_adapter','base','rows','feature_inputs',
                 'features','model_study','study'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    p.add_argument('--artifact-sha256',required=True);p.add_argument('--reference-sha256',required=True)
    a=p.parse_args()
    if a.study.exists():raise ValueError('Refusing to overwrite runtime verification')
    report=json.loads((a.model_study/'report.json').read_text())
    selection=json.loads((a.model_study/'selection.json').read_text())
    model_protocol=json.loads((a.model_study/'protocol.json').read_text())
    if (not report['complete'] or not selection['complete'] or report['selected']!=selection['selected']
            or file_sha(a.model_study/'selection.json')!=report['selection_sha256']
            or file_sha(a.weights)!=report['selected']['head_sha256']
            or file_sha(a.features)!=selection['features_sha256']
            or file_sha(a.feature_inputs)!=file_sha(a.model_study/'feature-inputs.jsonl')
            or model_protocol['prompt']!=runtime.PROMPT):raise ValueError('Frozen reference binding differs')
    rows=read_jsonl(a.rows);identities=read_jsonl(a.feature_inputs);indices=select_probes(rows,identities)
    a.study.mkdir(parents=True)
    protocol={'format':'roguard-response-runtime-parity-1','source_sha256':file_sha(__file__),
              'runtime_sha256':file_sha(runtime.__file__),'artifact_sha256':file_sha(a.artifact),
              'weights_sha256':file_sha(a.weights),'rows_sha256':file_sha(a.rows),
              'features_sha256':file_sha(a.features),'feature_inputs_sha256':file_sha(a.feature_inputs),
              'selection_sha256':report['selection_sha256'],'probe_indices':indices,
              'test_opened':False,'stable_release_ready':False}
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file
    from training.train_neural_response_v10 import make_head,prediction_rows
    x=np.load(a.features,allow_pickle=False)
    if x.shape!=(len(rows),2560) or not np.isfinite(x).all():raise ValueError('Invalid reference features')
    head=make_head(report['selected']['configuration'])
    head.load_weights([(key,mx.array(value)) for key,value in load_file(a.weights).items()]);head.eval()
    expected=head(mx.array(x[indices]));mx.eval(expected)
    expected=np.asarray(expected,dtype=np.float64);predictions=prediction_rows(expected)
    reference=Gemma4ReviewModel(a.reference_artifact,a.reference_adapter,a.base,expected_sha256=a.reference_sha256)
    model=Gemma4ResponseModel(a.artifact,a.weights,reference,expected_sha256=a.artifact_sha256)
    binary_payload={'language':reference.language,'turns':[{'role':'minor','text':'Discut despre un proiect obișnuit.' if reference.language=='ro' else 'Я говорю про звичайний проєкт.'}]}
    before=reference.predict(binary_payload);engine=reference._engine
    saved={key:np.asarray(value).copy() for key,value in tree_flatten(engine.model.trainable_parameters())}
    captured={};original_features=runtime._features
    def capture(current,data):
        value=original_features(current,data);captured['value']=value;return value
    runtime._features=capture;outputs=[];maximum_feature=0.;maximum_probability=0.;started=time.monotonic()
    try:
        for position,index in enumerate(indices):
            row=rows[index];result=model.review_response(row['draft'],row['language'],row['context'])
            if result['abstained']:raise ValueError('Supported reference probe abstained')
            feature_error=float(np.max(np.abs(captured['value'][0]-x[index])));maximum_feature=max(maximum_feature,feature_error)
            probability_error=max(abs(result['probabilities'][key]-predictions[position]['probabilities'][key]) for key in runtime.LABELS)
            maximum_probability=max(maximum_probability,probability_error)
            if feature_error>1e-3 or probability_error>1e-5 or result['label']!=predictions[position]['label']:
                raise ValueError('Installed response result differs from frozen training reference')
            outputs.append({'case_id':row['case_id'],'language':row['language'],'partition':row['partition'],
                            'input_sha256':identities[index]['input_sha256'],'result':result})
        import roguard.gemma4_attention as attention
        original_replay=attention.replay
        def failure(*args,**kwargs):raise RuntimeError('deliberate response runtime failure')
        attention.replay=failure
        try:
            try:model.review_response(rows[indices[0]]['draft'],rows[indices[0]]['language'],rows[indices[0]]['context'])
            except RuntimeError as exc:
                if str(exc)!='deliberate response runtime failure':raise
            else:raise ValueError('Failure injection did not reach inference')
        finally:attention.replay=original_replay
    finally:runtime._features=original_features
    after=reference.predict(binary_payload)
    if before!=after:raise ValueError('Response calls changed binary reference outputs')
    if any(not np.array_equal(np.asarray(value),saved[key]) for key,value in tree_flatten(engine.model.trainable_parameters())):
        raise ValueError('Response calls changed reference parameters')
    write_jsonl(a.study/'predictions.jsonl',outputs)
    result={'complete':True,'probes':len(indices),'all_selected_labels_match':True,
            'maximum_feature_error':maximum_feature,'maximum_probability_error':maximum_probability,
            'binary_reference_exact_after_success_and_failure':True,'all_reference_adapter_and_head_parameters_unchanged':True,
            'predictions_sha256':file_sha(a.study/'predictions.jsonl'),'seconds':time.monotonic()-started,
            'test_opened':False,'model_accuracy_qualification':False,'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
