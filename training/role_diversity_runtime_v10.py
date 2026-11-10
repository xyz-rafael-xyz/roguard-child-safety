"""Bindings and alignment for runtime references of the revised TRAIN corpus."""
import json
from pathlib import Path

from roguard.trajectory_data import file_sha,read_jsonl


def check_bindings(root,private,plan):
    for base,mapping in ((root,plan['public']),(private,plan['private'])):
        for name,sha in mapping.items():
            if file_sha(base/name)!=sha:raise ValueError('Registered input/source differs: '+name)


def validate_array(array,shape):
    import numpy as np
    if array.shape!=shape or array.dtype!=np.float32 or not np.isfinite(array).all():
        raise ValueError('Reference shape, dtype or finite values differ')


def align_development(recovered,metadata,language):
    selected=[r for r in recovered if r['language']==language and r['partition']=='dev']
    by_id={r['case_id']:r for r in selected}
    if len(by_id)!=len(selected) or len(metadata)!=len(selected):
        raise ValueError('Development population differs')
    output=[];seen=set()
    for row in metadata:
        key=row['case_id']
        if key not in by_id or key in seen:raise ValueError('Development row missing or repeated')
        seen.add(key);actual=by_id[key]
        if (row['partition']!='dev' or row['language']!=language
                or any(row[k]!=actual[k] for k in ('fields','gold','group','token_sha256'))):
            raise ValueError('Development alignment differs')
        output.append(actual)
    return output


def validate_specs(original,initial,v9_sha,initial_sha):
    if original['adapter_sha256']!=v9_sha or initial['adapter_sha256']!=initial_sha:
        raise ValueError('Specification weight identity differs')
    excluded={'adapter_sha256','mode','threshold'}
    if {k:v for k,v in original.items() if k not in excluded}!={k:v for k,v in initial.items() if k not in excluded}:
        raise ValueError('Initial runtime contract differs from v9')


def load_registered(root,private,plan_path,plan_sha):
    import numpy as np
    if file_sha(plan_path)!=plan_sha:raise ValueError('Reference plan differs')
    plan=json.loads(plan_path.read_text())
    if (plan['format']!='roguard-role-diversity-runtime-plan-1'
            or plan['rows']!=3168 or plan['development_rows']!=8837
            or plan['native_replay_tolerance']!=1e-3 or plan['reference_logit_tolerance']!=1e-5
            or plan['runtime_score_tolerance']!=1e-10 or plan['training_performed'] is not False):
        raise ValueError('Population, tolerances or reference-only scope differs')
    check_bindings(root,private,plan)
    proof100=json.loads((root/plan['verification100']).read_text())
    review100=json.loads((root/plan['author_review100']).read_text())
    if (proof100['verification']!='PASS' or proof100['rows']!=3168
            or proof100['rows_sha256']!=file_sha(private/plan['rows100'])
            or review100['rows_sha256']!=proof100['rows_sha256']
            or review100['decision']!='ELIGIBLE_FOR_SEPARATELY_REGISTERED_SYNTHETIC_TRAIN_EXPERIMENT'
            or proof100['training_admitted'] is not False or review100['training_admitted'] is not False):
        raise ValueError('Revised TRAIN corpus evidence differs')
    audit95=json.loads((root/plan['audit95']).read_text())
    report95=json.loads((root/plan['report95']).read_text())
    if (audit95['verification']!='PASS' or audit95['report_sha256']!=file_sha(root/plan['report95'])
            or not report95['complete'] or audit95['decision']!='REJECT' or audit95['model_qualified'] is not False):
        raise ValueError('Completed selected95 audit differs')
    rows=read_jsonl(private/plan['rows100'])
    if (len(rows)!=3168 or len({r['case_id'] for r in rows})!=3168
            or any(r['partition']!='train' or r['independently_annotated'] is not False for r in rows)):
        raise ValueError('Revised corpus is not the registered TRAIN-only population')
    recovered=read_jsonl(private/plan['recovered_inputs'])
    if any(r['partition'] not in ('train','dev') for r in recovered):raise ValueError('Reserved data encountered')
    output={}
    for lang,count in [('ro',4202),('uk',4635)]:
        config=plan['languages'][lang]
        fresh=[r for r in rows if r['language']==lang]
        if len(fresh)!=1584:raise ValueError('Language training population differs')
        metadata=read_jsonl(private/config['dev_rows'])
        dev=align_development(recovered,metadata,lang)
        if len(dev)!=count:raise ValueError('Language development population differs')
        refs=np.load(private/config['reference_logits'],allow_pickle=False);validate_array(refs,(count,19))
        weights={};specs={}
        for mode,base in [('v9',root),('selected95',private)]:
            model=config['models'][mode]
            weights[mode]=base/model['weights'];specs[mode]=json.loads((root/model['spec']).read_text())
            if file_sha(weights[mode])!=model['weights_sha256']:raise ValueError('Model weights differ')
        validate_specs(specs['v9'],specs['selected95'],file_sha(weights['v9']),file_sha(weights['selected95']))
        selected=audit95['selected'][lang]
        if (file_sha(weights['selected95'])!=selected['artifacts'][lang+'-selected.safetensors']
                or file_sha(private/config['reference_logits'])!=selected['artifacts'][lang+'-selected-reload-dev-logits.npy']
                or selected['epoch']!=report95['languages'][lang]['selected']['epoch']):
            raise ValueError('Selected checkpoint/reference differs from audit95')
        output[lang]=dict(train=fresh,dev=dev,reference=refs,weights=weights,specs=specs)
    return plan,output
