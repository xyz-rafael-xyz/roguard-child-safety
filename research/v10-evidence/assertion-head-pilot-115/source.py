"""Unregistered prototype: frozen-v9 features, matched data x assertion KL pilot."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


V9_WEIGHTS = {
    'ro': '114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
    'uk': '0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f',
}
NEW_PARTITIONS = {'fit': 48, 'probe_operator': 32, 'probe_meaning': 32,
                  'probe_combination': 16, 'probe_both': 16, 'counterfoil': 9}
FEATURE_BINDINGS = ('new_feature_audit', 'new_feature_auditor', 'new_feature_audit_plan',
                    'new_extraction_plan', 'new_feature_report') + tuple(
    language + '_' + suffix for language in ('ro', 'uk')
    for suffix in ('new_rows', 'new_features', 'new_logits', 'v9_weights'))


def object_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()


def validate_new_feature_audit(path):
    """Bind the completed CPU audit to exactly these frozen-v9 feature inputs."""
    audit = json.loads(path('new_feature_audit').read_text())
    if (audit.get('format') != 'roguard-assertion-probe-features-audit-1'
            or audit.get('verification') != 'PASS' or audit.get('device') != 'cpu'
            or any(audit.get(key) is not False for key in (
                'encoder_loaded', 'native_encoder_independently_replayed',
                'probe_accuracy_summaries_produced', 'training_performed',
                'training_admitted', 'held_out_test_opened', 'qualified'))):
        raise ValueError('Feature audit format, completion or scope differs')
    for field, key in (
            ('source_sha256', 'new_feature_auditor'),
            ('audit_plan_sha256', 'new_feature_audit_plan'),
            ('extraction_plan_sha256', 'new_extraction_plan'),
            ('producer_report_sha256', 'new_feature_report')):
        if audit.get(field) != sha(path(key)):
            raise ValueError('Feature audit dependency binding differs: ' + field)
    if any(type(audit.get(key)) is not int or audit[key] != 306
           for key in ('token_rows_replayed', 'head_rows_replayed')):
        raise ValueError('Feature audit replay population differs')
    artifacts = {language + '-' + name: sha(path(language + '_new_' + suffix))
                 for language in ('ro', 'uk')
                 for name, suffix in (('rows.jsonl', 'rows'), ('features.npy', 'features'), ('logits.npy', 'logits'))}
    if audit.get('feature_artifacts') != artifacts:
        raise ValueError('Feature audit artifact inventory or bytes differ')
    if audit.get('encoder_weights') != V9_WEIGHTS:
        raise ValueError('Feature audit original-v9 weights differ')
    if any(sha(path(language + '_v9_weights')) != V9_WEIGHTS[language]
           for language in ('ro', 'uk')):
        raise ValueError('Registered teacher is not original-v9')
    if not isinstance(audit.get('languages'), dict) or set(audit['languages']) != {'ro', 'uk'}:
        raise ValueError('Feature audit language inventory differs')
    for language in ('ro', 'uk'):
        summary = audit['languages'][language]
        if not isinstance(summary, dict):
            raise ValueError('Feature audit language summary differs')
        rows = read_rows(path(language + '_new_rows'))
        counts = dict(Counter(row['partition'] for row in rows))
        if (len(rows) != 153 or any(row['language'] != language for row in rows)
                or len({row['case_id'] for row in rows}) != 153
                or counts != NEW_PARTITIONS or summary.get('partitions') != NEW_PARTITIONS
                or any(type(summary.get(key)) is not int or summary[key] != expected
                       for key, expected in (('rows', 153), ('native_observations_recounted', 153),
                           ('seven_head_argmax_rows_differing', 0), ('binary_decision_differences', 0)))
                or summary.get('native_encoder_independently_replayed') is not False):
            raise ValueError('Feature audit per-language population or exact parity differs')
        order = [[row['case_id'], row['source_row_sha256'], row['payload_sha256'], row['token_sha256']]
                 for row in rows]
        if summary.get('ordered_inputs_sha256') != object_digest(order):
            raise ValueError('Feature audit ordered input binding differs')
        for key, limit in (('maximum_cpu_logit_error', 1e-3),
                           ('producer_native_maximum_recounted', 1e-3),
                           ('producer_score_maximum_recounted', 1e-10)):
            value = summary.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                raise ValueError('Feature audit numerical parity differs: ' + key)
    return audit


def matched_indices(old, new, repeat):
    if type(repeat) is not int or repeat < 1:
        raise ValueError('Invalid repetition count')
    pools = defaultdict(list)
    for i, row in enumerate(old):
        if row['partition'] == 'train':
            pools[(row['language'], tuple(row['fields']), row['gold'])].append(i)
        elif row['partition'] != 'dev':
            raise ValueError('Unexpected original partition')
    for pool in pools.values():
        pool.sort(key=lambda i: old[i]['case_id'])
    seen, counter, added, parents = set(), defaultdict(int), [], []
    for i, row in enumerate(new):
        if row['case_id'] in seen: raise ValueError('Duplicate new case identity')
        seen.add(row['case_id'])
        if row['partition'] != 'fit':
            if row['partition'] not in ('probe_operator','probe_meaning','probe_combination','probe_both','counterfoil'):
                raise ValueError('Unexpected new partition')
            continue
        key = row['language'], tuple(row['proposed_fields']), row['proposed_gold']
        if not pools[key]: raise ValueError('No matching original TRAIN parent')
        for _ in range(repeat):
            parents.append(pools[key][counter[key] % len(pools[key])])
            added.append(i); counter[key] += 1
    if not added: raise ValueError('No new fitting rows')
    return added, parents


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight-report',type=Path,help='Write exclusive preflight evidence and return before fitting')
    a=p.parse_args()
    if sha(a.plan)!=a.plan_sha256:raise ValueError('Plan differs')
    plan=json.loads(a.plan.read_text())
    if (plan['format']!='roguard-assertion-head-pilot-1' or plan['registered'] is not True
            or plan['synthetic_pilot_admitted'] is not True or plan['held_out_test_opened'] is not False):
        raise ValueError('No registered synthetic pilot admission')
    if (plan['epochs'],plan['seed'],plan['batch_size'],plan['repeat'])!=(12,1011,64,32):
        raise ValueError('Prospective schedule differs')
    if sha(__file__)!=plan['source_sha256']:raise ValueError('Executing trainer differs')
    def validate_files():
        if sha(a.plan)!=a.plan_sha256:raise ValueError('Plan changed')
        if sha(__file__)!=plan['source_sha256']:raise ValueError('Executing trainer changed')
        for record in plan['files'].values():
            if sha(record['path'])!=record['sha256']:raise ValueError('Input or source changed: '+record['path'])
    def validate_source_origins():
        root=Path(plan['project_root']).resolve()
        bound={Path(v['path']).resolve() for v in plan['files'].values()}
        for name,module in list(sys.modules.items()):
            if name.split('.')[0] not in ('roguard','training','eval','scripts') or not getattr(module,'__file__',None):continue
            actual=Path(module.__file__).resolve()
            base=root/'src' if name.split('.')[0]=='roguard' else root
            expected=base.joinpath(*name.split('.'))
            choices={expected.with_suffix('.py').resolve(),(expected/'__init__.py').resolve()}
            if actual not in choices or actual not in bound:
                raise ValueError('Unbound or external imported project module: '+name)
    validate_files()
    path=lambda key:Path(plan['files'][key]['path'])
    # The final registration must bind an actual completed independent feature audit.
    audit=validate_new_feature_audit(path)
    if a.preflight_report is not None:
        if a.preflight_report.exists():
            raise ValueError('Refusing existing preflight report')
        if a.preflight_report.resolve().is_relative_to(a.output.resolve()):
            raise ValueError('Preflight report must remain outside fit output')
    if a.output.exists():raise ValueError('Refusing overwrite')
    if shutil.disk_usage(a.output.parent).free<8*1024**3:raise ValueError('Disk reserve')
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file,save_file
    from training.assertion_teacher_loss_v10 import training_teacher_mask,teacher_kl_per_row
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass,policy_binary_cross_entropy
    from training.dropout_readout_v10 import make_head
    from training.prepare_runtime_warm_start_v10 import check_weights,compare_logits
    from training.native_fields_v9 import SIZES
    from roguard.gemma4_review import _score
    validate_source_origins()
    mx.set_default_device(mx.cpu)
    # Both languages must pass every preflight before the first optimizer exists.
    all_language_preflight={}
    for language in ('ro','uk'):
        before=read_rows(path(language+'_original_rows'));new_before=read_rows(path(language+'_new_rows'))
        if any(r['language']!=language for r in before+new_before):raise ValueError('Preflight language mismatch')
        for records,field_key,gold_key in ((before,'fields','gold'),(new_before,'proposed_fields','proposed_gold')):
            identities=set()
            for row in records:
                fields_before=row.get(field_key);gold_before=row.get(gold_key);identity=row.get('case_id')
                if (not isinstance(identity,str) or not identity or identity in identities
                        or not isinstance(fields_before,list) or len(fields_before)!=len(SIZES)
                        or any(type(v) is not int or not 0<=v<size for v,size in zip(fields_before,SIZES))
                        or type(gold_before) is not int or gold_before not in (0,1)):
                    raise ValueError('Preflight field/gold/identity metadata differs')
                identities.add(identity)
        if len(new_before)!=153 or sum(r['partition']=='fit' for r in new_before)!=48:
            raise ValueError('Preflight candidate population differs')
        fitting,controls=matched_indices(before,new_before,plan['repeat'])
        if sorted({new_before[i]['case_id'] for i in fitting})!=sorted(plan['admitted_fit_case_ids'][language]):
            raise ValueError('Preflight admitted fitting membership differs')
        train_before=[i for i,r in enumerate(before) if r['partition']=='train']
        development_before=[i for i,r in enumerate(before) if r['partition']=='dev']
        if (len(train_before),len(development_before))!=((13174,4202) if language=='ro' else (15755,4635)):
            raise ValueError('Preflight original population differs')
        loaded={}
        for suffix,shape in (('original_features',(len(before),2560)),('original_logits',(len(before),19)),
                             ('new_features',(len(new_before),2560)),('new_logits',(len(new_before),19))):
            values=np.load(path(language+'_'+suffix),mmap_mode='r',allow_pickle=False)
            if values.shape!=shape or values.dtype!=np.float32 or not np.isfinite(values).all():
                raise ValueError('Preflight feature shape/dtype/values differ')
            loaded[suffix]=values
        parent=load_file(path(language+'_v9_weights'));initial_before=load_file(path(language+'_initial_weights'))
        check_weights(parent,initial_before,language)
        initial_spec= json.loads(path(language+'_initial_spec').read_text());parent_spec=json.loads(path(language+'_v9_spec').read_text())
        excluded={'adapter_sha256','mode','threshold'}
        if (initial_spec['adapter_sha256']!=sha(path(language+'_initial_weights'))
                or parent_spec['adapter_sha256']!=sha(path(language+'_v9_weights'))
                or initial_spec['language']!=language or initial_spec['mode']!='coupled_policy'
                or initial_spec['threshold']!=(.7517333793124337 if language=='ro' else .9274248149759536)
                or {k:v for k,v in initial_spec.items() if k not in excluded}!={k:v for k,v in parent_spec.items() if k not in excluded}):
            raise ValueError('Preflight serving contract differs')
        mapping_before={'0.weight':'first.weight','0.bias':'first.bias','3.weight':'last.weight','3.bias':'last.bias'}
        head_before=make_head({k:initial_before['heads.'+language+'.'+v] for k,v in mapping_before.items()},0.)
        head_before.eval();parts_before=[]
        for start in range(0,len(development_before),256):
            z_before=head_before(mx.array(loaded['original_features'][development_before[start:start+256]]))
            mx.eval(z_before);parts_before.append(np.asarray(z_before).copy())
        reference_before=np.load(path(language+'_initial_dev_logits'),allow_pickle=False)
        all_language_preflight[language]=compare_logits(np.concatenate(parts_before),reference_before,initial_spec,limit=1e-3,exact_decisions=True)
        del head_before,loaded,parent,initial_before,parts_before,z_before,reference_before
        gc.collect();mx.clear_cache()
    validate_files();validate_source_origins()
    if a.preflight_report is not None:
        # Exclusive output preserves prior evidence and never creates the fit directory.
        preflight = dict(format='roguard-assertion-head-pilot-preflight-1', verification='PASS',
            plan_sha256=a.plan_sha256, source_sha256=sha(__file__),
            feature_audit_bindings={key:dict(plan['files'][key]) for key in FEATURE_BINDINGS},
            all_language_preflight=all_language_preflight, encoder_loaded=False,
            optimizer_started=False, training_performed=False, probe_scores_produced=False,
            probe_accuracy_summaries_produced=False,
            held_out_test_opened=False, qualified=False)
        with a.preflight_report.open('x') as handle:
            json.dump(preflight,handle,ensure_ascii=False,indent=2);handle.write('\n')
        return
    a.output.mkdir();write_json(a.output/'protocol.json',plan)
    write_json(a.output/'all-language-preflight.json',all_language_preflight)
    report=dict(complete=False,plan_sha256=a.plan_sha256,languages={},encoder_loaded=False,
                held_out_test_opened=False,qualified=False)
    started=time.monotonic()
    def persist():
        report['seconds']=time.monotonic()-started
        temp=a.output/'report.tmp';write_json(temp,report);temp.replace(a.output/'report.json')
    retained={}
    for lang in ('ro','uk'):
        old=read_rows(path(lang+'_original_rows'));new=read_rows(path(lang+'_new_rows'))
        if any(r['language']!=lang for r in old+new):raise ValueError('Language mismatch')
        original_x=np.load(path(lang+'_original_features'),allow_pickle=False)
        original_z=np.load(path(lang+'_original_logits'),allow_pickle=False)
        new_x=np.load(path(lang+'_new_features'),allow_pickle=False)
        new_z=np.load(path(lang+'_new_logits'),allow_pickle=False)
        for values,shape in ((original_x,(len(old),2560)),(original_z,(len(old),19)),(new_x,(len(new),2560)),(new_z,(len(new),19))):
            if values.shape!=shape or values.dtype!=np.float32 or not np.isfinite(values).all():raise ValueError('Feature shape/dtype/value differs')
        added,parents=matched_indices(old,new,plan['repeat'])
        pairs=[dict(occurrence=i,new_index=n,new_case_id=new[n]['case_id'],
                    parent_index=o,parent_case_id=old[o]['case_id'],fields=new[n]['proposed_fields'],gold=new[n]['proposed_gold'])
               for i,(n,o) in enumerate(zip(added,parents))]
        write_json(a.output/(lang+'-paired-occurrences.json'),pairs)
        expected_ids=plan['admitted_fit_case_ids'][lang]
        if sorted({new[i]['case_id'] for i in added})!=sorted(expected_ids) or len(expected_ids)!=48:
            raise ValueError('Admitted fit membership differs')
        train=np.array([i for i,r in enumerate(old) if r['partition']=='train'])
        dev=np.array([i for i,r in enumerate(old) if r['partition']=='dev'])
        if (len(train),len(dev))!=((13174,4202) if lang=='ro' else (15755,4635)):raise ValueError('Original population differs')
        initial=load_file(path(lang+'_initial_weights'));original=load_file(path(lang+'_v9_weights'))
        check_weights(original,initial,lang)
        spec=json.loads(path(lang+'_initial_spec').read_text());v9spec=json.loads(path(lang+'_v9_spec').read_text())
        if spec['adapter_sha256']!=sha(path(lang+'_initial_weights')) or v9spec['adapter_sha256']!=sha(path(lang+'_v9_weights')):
            raise ValueError('Specification weights differ')
        excluded={'adapter_sha256','mode','threshold'}
        if (spec['language']!=lang or spec['mode']!='coupled_policy'
                or spec['threshold']!=(.7517333793124337 if lang=='ro' else .9274248149759536)
                or {k:v for k,v in spec.items() if k not in excluded}!={k:v for k,v in v9spec.items() if k not in excluded}):
            raise ValueError('Initial serving contract differs')
        mapping={'0.weight':'first.weight','0.bias':'first.bias','3.weight':'last.weight','3.bias':'last.bias'}
        init={k:initial['heads.'+lang+'.'+v] for k,v in mapping.items()}
        dev_fields=np.array([old[i]['fields'] for i in dev],dtype=np.int32)
        dev_gold=np.array([old[i]['gold'] for i in dev],dtype=np.int32)
        v9_decisions=np.array([int(_score(z,v9spec)>=v9spec['threshold']) for z in original_z[dev]])
        reference=np.load(path(lang+'_initial_dev_logits'),allow_pickle=False)
        initial_decisions=np.array([int(_score(z,spec)>=spec['threshold']) for z in reference])
        policy=make_coupled_policy_mass(mx,spec['couplings'])
        rec=dict(training_unique_original=len(train),additional_unique=48,additional_occurrences=len(added),arms={})
        report['languages'][lang]=rec
        retained[lang]=(new,new_x,new_z,spec,v9spec)
        def project(head,x):
            head.eval();parts=[]
            for start in range(0,len(x),256):
                z=head(mx.array(x[start:start+256]));mx.eval(z);parts.append(np.asarray(z).copy())
            head.train();return np.concatenate(parts)
        def metrics(z,fields,gold,reference_decisions=None):
            pieces=np.split(z[:,:17].astype(np.float64),np.cumsum(SIZES)[:-1],axis=1)
            predicted=np.column_stack([part.argmax(axis=1) for part in pieces])
            decisions=np.array([int(_score(row,spec)>=spec['threshold']) for row in z])
            ce=[]
            for j,part in enumerate(pieces):
                exp=np.exp(part-part.max(axis=1,keepdims=True));prob=exp/exp.sum(axis=1,keepdims=True)
                ce.append(float(-np.log(np.maximum(prob[np.arange(len(z)),fields[:,j]],1e-30)).mean()))
            m=dict(rows=len(z),binary_errors=int((decisions!=gold).sum()),joint_errors=int(np.any(predicted!=fields,axis=1).sum()),field_errors=(predicted!=fields).sum(axis=0).tolist(),mean_field_ce=float(np.mean(ce)))
            if reference_decisions is not None:
                m.update(introduced_v9=int(((decisions!=gold)&(reference_decisions==gold)).sum()),fixed_v9=int(((decisions==gold)&(reference_decisions!=gold)).sum()))
            return m,decisions
        for data_mode in ('control','new'):
            extra_x,extra_z=(original_x[parents],original_z[parents]) if data_mode=='control' else (new_x[added],new_z[added])
            x=np.concatenate((original_x[train],extra_x));teacher=np.concatenate((original_z[train],extra_z))
            fit_rows=[old[i] for i in train]+[dict(partition='train',fields=new[i]['proposed_fields'],gold=new[i]['proposed_gold']) for i in added]
            fields=np.array([r['fields'] for r in fit_rows],dtype=np.int32);gold=np.array([r['gold'] for r in fit_rows],dtype=np.int32)
            weights=[]
            for j,size in enumerate(SIZES):
                counts=np.bincount(fields[:,j],minlength=size);w=np.minimum(4.,np.sqrt(len(fields)/np.maximum(1,counts)/size));w/=np.mean(w[fields[:,j]])
                weights.append(mx.array(w.astype(np.float32)))
            for distillation in ('full_teacher','mask_wrong_assertion'):
                arm=data_mode+'-'+distillation;directory=a.output/lang/arm;directory.mkdir(parents=True)
                mask=training_teacher_mask(fit_rows,teacher,distillation)
                np.save(directory/'assertion-keep.npy',mask,allow_pickle=False)
                record=dict(epochs=[],selected=None,mask_false=int((~mask).sum()),mask_sha256=sha(directory/'assertion-keep.npy'),
                            paired_occurrences_sha256=sha(a.output/(lang+'-paired-occurrences.json')),
                            train_fields_sha256=hashlib.sha256(fields.tobytes()).hexdigest(),
                            train_gold_sha256=hashlib.sha256(gold.tobytes()).hexdigest(),
                            train_teacher_sha256=hashlib.sha256(teacher.tobytes()).hexdigest(),
                            train_features_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                            array_digest_contract='C-contiguous float32 features/teacher, int32 fields/gold; shapes follow registered populations')
                rec['arms'][arm]=record
                mx.random.seed(1011);rng=np.random.default_rng(1011)
                head=make_head(init,.1 if lang=='ro' else 0.);mx.eval(head.parameters())
                opt=optim.AdamW(learning_rate=.0001,weight_decay=.01)
                def loss_fn(model,fx,y,fy,tz,keep):
                    z=model(fx);offset=0;terms=[]
                    for j,size in enumerate(SIZES):
                        terms.append(nn.losses.cross_entropy(z[:,offset:offset+size],fy[:,j])*weights[j][fy[:,j]]);offset+=size
                    return (nn.losses.cross_entropy(z[:,-2:],y)+mx.stack(terms).mean(axis=0)+.2*teacher_kl_per_row(mx,z,tz,keep)+(.25 if lang=='ro' else 1.)*policy_binary_cross_entropy(mx,policy(z[:,:17]),y)).mean()
                value_grad=nn.value_and_grad(head,loss_fn)
                for epoch in range(13):
                    if epoch:
                        losses=[];order=rng.permutation(len(x))
                        for start in range(0,len(x),64):
                            ix=order[start:start+64]
                            loss,gradient=value_grad(head,mx.array(x[ix]),mx.array(gold[ix]),mx.array(fields[ix]),mx.array(teacher[ix]),mx.array(mask[ix]))
                            gradient,norm=optim.clip_grad_norm(gradient,1.);mx.eval(loss,norm,gradient)
                            if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(norm).item()):raise ValueError('Nonfinite optimizer step')
                            opt.update(head,gradient);mx.eval(head.parameters(),opt.state);losses.append(float(loss.item()))
                    z=project(head,original_x[dev])
                    if epoch==0:compare_logits(z,reference,spec,limit=1e-3,exact_decisions=True)
                    m,decisions=metrics(z,dev_fields,dev_gold,v9_decisions)
                    m.update(epoch=epoch,updates=epoch*((len(x)+63)//64),presentations=epoch*len(x),introduced_initial75=int(((decisions!=dev_gold)&(initial_decisions==dev_gold)).sum()))
                    if epoch:m['training_loss']=float(np.mean(losses))
                    checkpoint=directory/f'epoch-{epoch}.safetensors';zp=directory/f'epoch-{epoch}-dev.npy'
                    save_file({k:np.asarray(v).copy() for k,v in tree_flatten(head.parameters())},str(checkpoint));np.save(zp,z,allow_pickle=False)
                    m.update(checkpoint_sha256=sha(checkpoint),dev_logits_sha256=sha(zp),rank=[m['binary_errors'],m['joint_errors'],m['mean_field_ce'],epoch])
                    record['epochs'].append(m)
                    if record['selected'] is None or m['rank']<record['selected']['rank']:record['selected']=dict(m)
                    persist();print(json.dumps(dict(language=lang,arm=arm,**m)),flush=True)
                    if shutil.disk_usage(a.output).free<8*1024**3:raise ValueError('Disk reserve')
                del head,opt,value_grad;gc.collect();mx.clear_cache()
    validate_files()
    validate_source_origins()
    frozen={lang:{arm:rec['selected'] for arm,rec in info['arms'].items()} for lang,info in report['languages'].items()}
    write_json(a.output/'selected-before-probes.json',frozen)
    report.update(fitting_complete=True,selection_sha256=sha(a.output/'selected-before-probes.json'),probe_scoring_started=False)
    persist()
    # Deliberately separate probe evaluation into the independently reviewed audit.
    # This prototype does not yet implement that evaluator and is not registered.
    report.update(complete=True,decision='FIT_COMPLETE_REQUIRES_INDEPENDENT_REPLAY_AND_FROZEN_PROBE_EVALUATION')
    persist()


if __name__=='__main__':main()
