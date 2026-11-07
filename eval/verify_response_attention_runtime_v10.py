"""Verify installed attention response inference against frozen DEV and full forwards."""
import argparse
import json
from pathlib import Path
import time

import roguard.gemma4_response_attention as runtime
from roguard.assurance import digest
from roguard.gemma4_review import Gemma4ReviewModel
from roguard.trajectory_data import file_sha,read_jsonl,write_jsonl


def development_rows(rows,identities):
    if len(rows)!=len(identities):raise ValueError('Incomplete input mapping')
    result=[]
    for row,identity in zip(rows,identities):
        turns=([{'role':'minor','text':row['context']}] if row['context'] else [])+[{'role':'assistant','text':row['draft']}]
        if identity!={'case_id':row['case_id'],'partition':row['partition'],
                      'input_sha256':digest({'language':row['language'],'turns':turns})}:
            raise ValueError('Response input mapping differs')
        if row['partition'] not in ('train','dev'):raise ValueError('Reserved inputs are prohibited')
        if row['partition']=='dev':result.append(row)
    if not result or len({r['case_id'] for r in result})!=len(result):raise ValueError('Empty or duplicate DEV mapping')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact','weights','reference_artifact','reference_adapter','base','rows','model_study','model_private','study'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    p.add_argument('--artifact-sha256',required=True);p.add_argument('--reference-sha256',required=True)
    a=p.parse_args()
    if a.study.exists():raise ValueError('Refusing to overwrite attention runtime verification')
    report=json.loads((a.model_study/'report.json').read_text())
    selection=json.loads((a.model_study/'selection.json').read_text())
    spec=json.loads((a.model_study/'model.json').read_text())
    if (report.get('complete') is not True or selection.get('complete') is not True
            or report['selected']!=selection['selected']
            or report['selection_sha256']!=file_sha(a.model_study/'selection.json')
            or report['model_spec_sha256']!=file_sha(a.model_study/'model.json')
            or spec['protocol_sha256']!=file_sha(a.model_study/'protocol.json')
            or file_sha(a.weights)!=report['selected']['adapter_sha256']
            or file_sha(a.weights)!=spec['adapter_sha256']):raise ValueError('Selected attention reference differs')
    rows=development_rows(read_jsonl(a.rows),read_jsonl(a.model_study/'feature-inputs.jsonl'))
    logits_path=a.model_private/'selected-development-logits.npy'
    a.study.mkdir(parents=True)
    protocol={'format':'roguard-response-attention-runtime-parity-1','source_sha256':file_sha(__file__),
              'runtime_sha256':file_sha(runtime.__file__),'artifact_sha256':file_sha(a.artifact),
              'weights_sha256':file_sha(a.weights),'rows_sha256':file_sha(a.rows),
              'reference_logits_sha256':file_sha(logits_path),'selection_sha256':report['selection_sha256'],
              'model_spec_sha256':report['model_spec_sha256'],'development_rows':len(rows),
              'test_opened':False,'stable_release_ready':False}
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file
    from training.train_neural_response_v10 import make_head,prediction_rows,PROMPT
    from training.response_attention_v10 import make_response_tail,load_response_weights
    from training.probe_assertion_query_v10 import tokens_for
    if spec['prompt']!=PROMPT:raise ValueError('Reference query differs')
    expected=np.load(logits_path,allow_pickle=False)
    if expected.shape!=(len(rows),5) or not np.isfinite(expected).all():raise ValueError('Invalid reference logits')
    predictions=prediction_rows(expected.astype(np.float64))
    reference=Gemma4ReviewModel(a.reference_artifact,a.reference_adapter,a.base,expected_sha256=a.reference_sha256)
    model=runtime.Gemma4ResponseAttentionModel(a.artifact,a.weights,reference,expected_sha256=a.artifact_sha256)
    text='Un proiect obișnuit.' if reference.language=='ro' else 'Звичайний проєкт.'
    binary_payload={'language':reference.language,'turns':[{'role':'minor','text':text}]}
    before=reference.predict(binary_payload);engine=reference._engine
    saved=dict(tree_flatten(engine.model.trainable_parameters()))
    original_bytes={key:np.asarray(value).copy() for key,value in saved.items()}
    started=time.monotonic();outputs=[];maximum_probability=0.;maximum_full=0.
    for i,(row,prediction) in enumerate(zip(rows,predictions)):
        result=model.review_response(row['draft'],row['language'],row['context'])
        if result['abstained'] or result['label']!=prediction['label']:raise ValueError('Installed attention decision differs')
        error=max(abs(result['probabilities'][label]-prediction['probabilities'][label]) for label in runtime.LABELS)
        maximum_probability=max(maximum_probability,error)
        if error>1e-5:raise ValueError('Installed attention probabilities differ')
        outputs.append({'case_id':row['case_id'],'result':result})
        if (i+1)%32==0:print(json.dumps({'phase':'installed','completed':i+1,'total':len(rows)}),flush=True)
    # Independently reload the selected training module, then use full forwards,
    # rather than the packaged prefix/replay inference path.
    training_model=make_response_tail(engine.model.base,make_head(spec['head']))
    try:
        load_response_weights(training_model,load_file(a.weights));training_model.eval()
        for i,row in enumerate(rows):
            payload={'language':row['language'],'turns':([{'role':'minor','text':row['context']}] if row['context'] else [])+
                     [{'role':'assistant','text':row['draft']}]}
            tokens,_=tokens_for(engine.tokenizer,PROMPT,payload)
            if not tokens or len(tokens)>1024:raise ValueError('Reference input exceeds envelope')
            values=training_model.head(training_model.base.model(mx.array([tokens]))[:,-1,:]);mx.eval(values)
            actual=np.asarray(values)[0];error=float(np.max(np.abs(actual-expected[i])));maximum_full=max(maximum_full,error)
            if error>1e-3 or int(actual.argmax())!=int(expected[i].argmax()):raise ValueError('Full-forward reference differs')
            mx.clear_cache()
            if (i+1)%32==0:print(json.dumps({'phase':'full-forward','completed':i+1,'total':len(rows)}),flush=True)
    finally:
        engine.model.load_weights(list(saved.items()),strict=False);mx.eval(engine.model.trainable_parameters());mx.clear_cache()
    import roguard.gemma4_attention as attention
    original_replay=attention.replay
    def failure(*args,**kwargs):raise RuntimeError('deliberate attention runtime failure')
    attention.replay=failure
    try:
        try:model.review_response(rows[0]['draft'],rows[0]['language'],rows[0]['context'])
        except RuntimeError as exc:
            if str(exc)!='deliberate attention runtime failure':raise
        else:raise ValueError('Failure injection did not reach inference')
    finally:attention.replay=original_replay
    after=reference.predict(binary_payload)
    unchanged=all(np.array_equal(np.asarray(value),original_bytes[key]) for key,value in tree_flatten(engine.model.trainable_parameters()))
    if before!=after or not unchanged:raise ValueError('Attention response calls altered binary reference')
    write_jsonl(a.study/'predictions.jsonl',outputs)
    result={'complete':True,'development_rows':len(rows),'all_selected_labels_match':True,
            'maximum_probability_error':maximum_probability,'maximum_full_forward_logit_error':maximum_full,
            'binary_reference_exact_after_success_and_failure':True,'all_reference_adapter_and_head_parameters_unchanged':True,
            'predictions_sha256':file_sha(a.study/'predictions.jsonl'),'seconds':time.monotonic()-started,
            'test_opened':False,'model_accuracy_qualification':False,'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
