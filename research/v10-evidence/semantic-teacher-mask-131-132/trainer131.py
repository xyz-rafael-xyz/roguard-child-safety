"""Unregistered two-arm new-FIT semantic disagreement-mask ablation on frozen-v9 features."""
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
    raw=(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    Path(path).write_bytes(raw)
    if Path(path).read_bytes()!=raw:raise ValueError('JSON write/readback differs')


V9_WEIGHTS = {
    'ro': '114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
    'uk': '0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f',
}
NEW_PARTITIONS = {'fit':288,'probe_operator':384,'probe_meaning':384,
                  'probe_combination':96,'probe_both':384,'counterfoil':4}
FEATURE_BINDINGS = ('new_feature_audit', 'new_feature_auditor', 'new_feature_audit_plan',
                    'new_extraction_plan', 'new_feature_report') + tuple(
    language + '_' + suffix for language in ('ro', 'uk')
    for suffix in ('new_rows', 'new_features', 'new_logits', 'v9_weights'))


def object_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()


INITIAL75 = {'ro':'1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3',
             'uk':'466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}
INITIAL_SPECS = {'ro':'5fdeae97843d873755ed41a02453b5c29823fd8028508c3e99d1586175c8f4a9',
                 'uk':'bbf3fea8ee298c40d86701851fe045352ea1a924ba5287dd52b70e59ce25de27'}
CONFIGURATION = dict(learning_rate=.0001,weight_decay=.01,gradient_clip=1.,
    dropout=dict(ro=.1,uk=0.),teacher_kl=.2,policy_ce=dict(ro=.25,uk=1.))
ARMS=('new-mask_wrong_assertion','new-mask_wrong_new_fit_semantics')
GATE_CONTRACT = dict(primary='new-mask_wrong_new_fit_semantics',control='new-mask_wrong_assertion',
    strict_assertion_partitions=['probe_operator','probe_both'],strict_rows_per_language=768,
    strict_references=['new-mask_wrong_assertion','frozen125-primary','frozen125-control-full_teacher','unchanged-initial75','original-v9'],
    strict_frozen125_primary_assertion_errors=dict(ro=106,uk=102),
    probe_nonregression_partitions=['probe_operator','probe_meaning','probe_combination','probe_both'],
    probe_nonregression_metrics=['binary_errors','joint_errors','assertion_errors'],
    probe_nonregression_references=['new-mask_wrong_assertion','frozen125-primary','frozen125-control-full_teacher'],
    original_dev_nonregression_metrics=['binary_errors','joint_errors','introduced_v9'],
    original_dev_nonregression_references=['new-mask_wrong_assertion','frozen125-primary','frozen125-control-full_teacher','unchanged-initial75'],
    fixed_original_dev_ceilings=dict(ro=dict(binary_errors=12,joint_errors=673,introduced_v9=2),uk=dict(binary_errors=1,joint_errors=634,introduced_v9=0)),
    control_reproduction_required=True,control_reproduction_tolerance=.001,
    panel_rows=40,panel_alias_rows=32,panel_unique_supplemental_rows=8,
    panel_binary_errors=0,panel_joint_errors=0,secondary_cannot_replace_primary=True,
    independent_human_validation=False,qualification=False)


def derived_budget():
    populations={language:original+288*32 for language,original in [('ro',13174),('uk',15755)]}
    updates={language:12*((n+63)//64) for language,n in populations.items()}
    return dict(train_occurrences_per_language_per_arm=populations,
        updates_per_language_per_arm=updates,train_occurrences_per_arm=sum(populations.values()),
        updates_per_arm=sum(updates.values()),updates_all2arms=2*sum(updates.values()),
        presentations_per_arm=12*sum(populations.values()),presentations_all2arms=24*sum(populations.values()),
        development_rows=8837,new_unique_fit_inputs=576,new_core_propositions_per_language_in_fit=8,
        occurrences_are_not_independent_examples=True)


def validate_plan_contract(plan):
    for key,expected in [('epochs',12),('seed',1011),('batch_size',64),('repeat',32)]:
        if type(plan[key]) is not int or plan[key]!=expected:raise ValueError('Prospective schedule differs')
    if (object_digest(plan['configuration'])!=object_digest(CONFIGURATION)
            or object_digest(plan['gate_contract'])!=object_digest(GATE_CONTRACT)
            or object_digest(plan['scheduled_budget'])!=object_digest(derived_budget())
            or plan['new_probe_accuracy_scored_before_registration'] is not True
            or plan['new_trained_probe_scoring_before_selection_freeze'] is not False):
        raise ValueError('Fixed larger study configuration/gate/budget/exposure differs')
    for language in ('ro','uk'):
        if (plan['files'][language+'_initial_weights']['sha256']!=INITIAL75[language]
                or plan['files'][language+'_initial_spec']['sha256']!=INITIAL_SPECS[language]):
            raise ValueError('Only unchanged75 from90 and its exact spec may initialize')
        identities=plan['admitted_fit_case_ids'][language]
        if len(identities)!=288 or len(set(identities))!=288:raise ValueError('Exact288 fitting identities required')


def pointer(value,path):
    for part in path.strip('/').split('/'):value=value[part]
    return value


def validate_registered_lineage(plan,path):
    if len(plan['original_lineage'])!=3 or {item['role'] for item in plan['original_lineage']}!={'original88','audit89','warm90'}:
        raise ValueError('Original feature/head lineage inventory differs')
    original_keys={lang+'_'+key for lang in ('ro','uk') for key in ('original_rows','original_features','original_logits')}
    initial_keys={lang+'_'+key for lang in ('ro','uk') for key in ('initial_weights','initial_dev_logits','initial_dev_rows')}
    coverage={'original88':original_keys,'audit89':original_keys|{'original_feature_report'},'warm90':initial_keys}
    for item in plan['original_lineage']:
        statuses={'/verification':'PASS','/qualified':False} if item['role']=='audit89' else {'/complete':True,'/qualified':False}
        if (any(object_digest(item['expected_values'].get(k))!=object_digest(v) for k,v in statuses.items())
                or set(item['artifact_bindings'].values())!=coverage[item['role']]):
            raise ValueError('Mandatory original lineage completion/artifact coverage missing')
        record=json.loads(path(item['file_key']).read_text())
        for key,expected in item['expected_values'].items():
            if object_digest(pointer(record,key))!=object_digest(expected):raise ValueError('Original lineage status differs')
        for key,binding in item['artifact_bindings'].items():
            if pointer(record,key)!=sha(path(binding)):raise ValueError('Original lineage artifact differs')
    design=json.loads(path('prospective_design').read_text())
    if (sha(path('prospective_design'))!=plan['prospective_design_sha256']
            or object_digest(design['gate_contract'])!=object_digest(GATE_CONTRACT)):
        raise ValueError('Prospective design/gate binding differs')


def validate_baseline_and_aliases(plan,path):
    baseline=json.loads(path('baseline122_report').read_text())
    if (baseline['format']!='roguard-balanced-baseline-profile-1' or baseline['complete'] is not True
            or baseline['source_sha256']!=sha(path('baseline122_source'))
            or baseline['plan_sha256']!=sha(path('baseline122_plan'))
            or baseline['eligibility']['strict_improvement_study_eligible'] is not True
            or baseline['qualified'] is not False):raise ValueError('Completed eligible122 baseline required')
    for filename,h in baseline['artifacts'].items():
        if h!=sha(path('baseline122_artifact:'+filename)):raise ValueError('Bound122 prediction/artifact differs')
    initial122=json.loads(path('baseline122_plan').read_text())
    if initial122['files']['feature_audit121']['sha256']!=sha(path('new_feature_audit')):
        raise ValueError('Baseline122 and training feature audit differ')
    verification=json.loads(path('baseline122_verification').read_text())
    fixed={'/verification':'PASS','/rows_recounted':6160,'/panel_responses_recounted':80,
        '/initial75_head_rows_independently_replayed':3080,'/all_group_metrics_exact':True,
        '/all_panel_metrics_exact':True,'/encoder_loaded':False,'/training_performed':False,'/qualified':False}
    for key,value in fixed.items():
        if object_digest(pointer(verification,key))!=object_digest(value):raise ValueError('Mandatory independent122 verification differs')
    if (verification['producer_report_sha256']!=sha(path('baseline122_report'))
            or verification['source_sha256']!=sha(path('baseline122_verification_source'))
            or object_digest(verification['eligibility'])!=object_digest(baseline['eligibility'])):
        raise ValueError('Independent122 producer/source/eligibility differs')
    for language in ('ro','uk'):
        parity=verification['head_parity'][language]
        for key,value in [('rows',1540),('seven_head_argmax_differences',0),('binary_decision_differences',0)]:
            if type(parity[key]) is not int or parity[key]!=value:raise ValueError('Independent122 replay population/decisions differ')
        error=parity['max_cpu_logit_error']
        if type(error) not in (float,int) or not math.isfinite(error) or not 0<=error<=1e-3:
            raise ValueError('Independent122 CPU head parity differs')
    for filename,h in verification['artifacts'].items():
        if h!=sha(path('baseline122_verification_artifact:'+filename)):raise ValueError('Independent122 retained replay differs')
    required=plan['baseline122_verification_contract']
    if not required or not required['expected_values'] or not required['artifact_bindings']:
        raise ValueError('Independent122 verification contract missing')
    for key,expected in required['expected_values'].items():
        if object_digest(pointer(verification,key))!=object_digest(expected):raise ValueError('Independent122 verification status differs')
    for key,binding in required['artifact_bindings'].items():
        if pointer(verification,key)!=sha(path(binding)):raise ValueError('Independent122 verification input differs')
    for language,expected in [('ro',(343,357)),('uk',(274,296))]:
        for reference,n in zip(('original-v9','unchanged-initial75'),expected):
            group=baseline['languages'][language][reference]['primary_transfer']['all']
            if type(group['rows']) is not int or group['rows']!=768 or type(group['assertion_errors']) is not int or group['assertion_errors']!=n:
                raise ValueError('Frozen122 fixed transfer population/result differs')
    mapping=json.loads(path('full_panel_alias_map').read_text());panel=read_rows(path('full_panel'))
    candidates={row['case_id']:row for language in ('ro','uk') for row in read_rows(path(language+'_new_rows'))}
    if len(mapping['aliases'])!=40 or len(panel)!=40:raise ValueError('Full40 panel binding differs')
    partitions=Counter()
    for index,(alias,original) in enumerate(zip(mapping['aliases'],panel)):
        if type(alias['panel_index']) is not int or alias['panel_index']!=index or alias['panel_example_id']!=original['example_id']:
            raise ValueError('Full40 panel index/type differs')
        row=candidates[alias['canonical_case_id']];partitions[row['partition']]+=1
        if (object_digest(alias['proposed_fields'])!=object_digest(row['proposed_fields'])
                or object_digest(original['proposed_binary'])!=object_digest(row['proposed_gold'])):
            raise ValueError('Full40 panel alias target differs')
    if partitions!={'probe_operator':16,'probe_both':16,'counterfoil':8}:
        raise ValueError('Frozen panel fold membership differs; no fitting aliases permitted')
    audit=json.loads(path('new_feature_audit').read_text())
    if audit['full_panel_sha256']!=sha(path('full_panel')) or audit['panel_alias_map_sha256']!=sha(path('full_panel_alias_map')):
        raise ValueError('Audit121 full40 panel identity differs')


def validate_frozen125_reference(plan,path):
    fit=json.loads(path('frozen125_fit_report').read_text());evaluation=json.loads(path('frozen126_report').read_text())
    if (fit.get('complete') is not True or fit.get('qualified') is not False
            or fit['plan_sha256']!=sha(path('frozen125_plan'))
            or fit['source_sha256']!=sha(path('frozen125_source'))
            or evaluation.get('verification')!='PASS' or evaluation.get('decision')!='REJECT_ESCALATION'
            or evaluation['fit_report_sha256']!=sha(path('frozen125_fit_report'))
            or evaluation['source_sha256']!=sha(path('frozen126_source'))
            or evaluation['evaluation_plan_sha256']!=sha(path('frozen126_plan'))):
        raise ValueError('Frozen125/126 reference lineage differs')
    for lang,expected in [('ro',106),('uk',102)]:
        results=evaluation['results'][lang];primary=results['arms']['new-mask_wrong_assertion']
        count=sum(primary['groups']['partition:'+part]['assertion_errors'] for part in ('probe_operator','probe_both'))
        if count!=expected:raise ValueError('Frozen125 primary transfer reference differs')
        floors={key:min(results['arms'][arm]['dev'][key] for arm in ('new-mask_wrong_assertion','control-full_teacher')) for key in ('binary_errors','joint_errors','introduced_v9')}
        floors={key:min(value,results['initial75_dev'][key]) for key,value in floors.items()}
        if object_digest(floors)!=object_digest(GATE_CONTRACT['fixed_original_dev_ceilings'][lang]):raise ValueError('Frozen originalDEV gate floor differs')
        epochs=fit['languages'][lang]['arms']['new-mask_wrong_assertion']['epochs']
        if len(epochs)!=13:raise ValueError('Frozen125 control epoch inventory differs')
        for i,epoch in enumerate(epochs):
            if epoch['epoch']!=i or epoch['dev_logits_sha256']!=sha(path(lang+'_frozen125_epoch_'+str(i)+'_dev')):
                raise ValueError('Frozen125 control DEV trajectory differs')


def validate_feature_norms(values,np):
    for start in range(0,len(values),256):
        with np.errstate(over='ignore',invalid='ignore'):
            norm=(values[start:start+256]*values[start:start+256]).sum(axis=1)
        if not np.isfinite(norm).all() or not (norm>0).all():raise ValueError('Nonfinite or zero feature normalization')


def verify_saved_mask(path,expected,np):
    actual=np.load(path,allow_pickle=False)
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or actual.tobytes()!=expected.tobytes():
        raise ValueError('Retained teacher mask differs before optimizer')


def finalize_output(output,report,persist,check_inputs,check_origins,check_artifacts,snapshot):
    report.update(complete=True,decision='FIT_COMPLETE_REQUIRES_INDEPENDENT_REPLAY_AND_FROZEN_PROBE_EVALUATION')
    try:
        persist()
        if object_digest(json.loads((output/'report.json').read_text()))!=object_digest(report):raise ValueError('Final report readback differs')
        check_inputs();check_origins();check_artifacts()
        if any(sha(output/name)!=digest for name,digest in snapshot.items()):raise ValueError('Retained artifact changed after final report')
        # Check again after every final output readback, before accepting completion.
        check_inputs();check_origins()
        if object_digest(json.loads((output/'report.json').read_text()))!=object_digest(report):raise ValueError('Final report changed at completion')
    except BaseException as error:
        final_path=output/'report.json'
        if final_path.exists():final_path.replace(output/'rejected-final-report.json')
        report.update(complete=False,decision='FIT_OUTPUT_INTEGRITY_FAILED',error_type=type(error).__name__,error=str(error))
        try:
            write_json(final_path,report)
            if object_digest(json.loads(final_path.read_text()))!=object_digest(report):raise ValueError('Incomplete report readback differs')
        except BaseException:
            if final_path.exists():final_path.replace(output/'unverified-incomplete-report.json')
        raise


def verify_saved_checkpoint(checkpoint,logits_path,tensors,expected,np,load_file):
    actual=load_file(str(checkpoint));z=np.load(logits_path,allow_pickle=False)
    if set(actual)!=set(tensors) or z.shape!=expected.shape or z.dtype!=expected.dtype or z.tobytes()!=expected.tobytes():
        raise ValueError('Saved checkpoint/DEV logits differ')
    for key,value in tensors.items():
        if actual[key].shape!=value.shape or actual[key].dtype!=value.dtype or actual[key].tobytes()!=value.tobytes():
            raise ValueError('Saved head tensor differs')


def verify_all_artifacts(output,report,plan,np,load_file):
    if (output/'source.py').read_bytes()!=Path(__file__).read_bytes() or object_digest(json.loads((output/'protocol.json').read_text()))!=object_digest(plan):
        raise ValueError('Retained source/protocol differs')
    if set(report['languages'])!={'ro','uk'}:raise ValueError('Both languages and both arms must freeze before any new trained probe')
    for language,info in report['languages'].items():
        if set(info['arms'])!=set(ARMS):
            raise ValueError('Complete two-arm inventory required')
        for arm,record in info['arms'].items():
            if (sha(output/language/arm/'assertion-keep.npy')!=record['mask_sha256']
                    or sha(output/(language+'-paired-occurrences.json'))!=record['paired_occurrences_sha256']):
                raise ValueError('Retained teacher mask or matched occurrences changed')
            if len(record['epochs'])!=13 or [v['epoch'] for v in record['epochs']]!=list(range(13)):
                raise ValueError('Complete epoch0 through12 retention required')
            final=record['epochs'][-1]
            budget=derived_budget()
            if final['updates']!=budget['updates_per_language_per_arm'][language] or final['presentations']!=12*budget['train_occurrences_per_language_per_arm'][language]:
                raise ValueError('Completed update/presentation budget differs')
            if object_digest(record['selected'])!=object_digest(min(record['epochs'],key=lambda v:v['rank'])):
                raise ValueError('OldDEV-only selected identity differs')
            for epoch in record['epochs']:
                folder=output/language/arm
                if sha(folder/f"epoch-{epoch['epoch']}.safetensors")!=epoch['checkpoint_sha256'] or sha(folder/f"epoch-{epoch['epoch']}-dev.npy")!=epoch['dev_logits_sha256']:
                    raise ValueError('Retained checkpoint changed before selection freeze')


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
    if any(type(audit.get(key)) is not int or audit[key] != 3080
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
        if (len(rows) != 1540 or any(row['language'] != language for row in rows)
                or len({row['case_id'] for row in rows}) != 1540
                or counts != NEW_PARTITIONS or summary.get('partitions') != NEW_PARTITIONS
                or any(type(summary.get(key)) is not int or summary[key] != expected
                       for key, expected in (('rows', 1540), ('native_observations_recounted', 1540),
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


def validate_dev_order(old,reference):
    development=[row for row in old if row['partition']=='dev']
    keys=('case_id','fields','gold','token_sha256')
    if object_digest([[r[k] for k in keys] for r in development])!=object_digest([[r[k] for k in keys] for r in reference]):
        raise ValueError('Original DEV metadata order differs from initial75 reference')


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
    if (plan['format']!='roguard-new-fit-semantic-mask-study-1' or plan['registered'] is not True
            or plan['synthetic_pilot_admitted'] is not True or plan['held_out_test_opened'] is not False):
        raise ValueError('No registered synthetic pilot admission')
    validate_plan_contract(plan)
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
    validate_registered_lineage(plan,path)
    validate_baseline_and_aliases(plan,path)
    validate_frozen125_reference(plan,path)
    if a.preflight_report is not None:
        if a.preflight_report.exists():
            raise ValueError('Refusing existing preflight report')
        if a.preflight_report.resolve().is_relative_to(a.output.resolve()):
            raise ValueError('Preflight report must remain outside fit output')
    if a.output.exists():raise ValueError('Refusing overwrite')
    if shutil.disk_usage(a.output.parent).free<8*1024**3:raise ValueError('Disk reserve')
    import importlib.metadata
    for package,expected in plan['numerical_versions'].items():
        if importlib.metadata.version(package)!=expected:raise ValueError('Numerical package version differs: '+package)
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from safetensors.numpy import load_file,save_file
    # Compile only the exact bound private loss bytes, before any numerical preflight.
    import types
    loss_raw=path('semantic_teacher_loss').read_bytes()
    if hashlib.sha256(loss_raw).hexdigest()!=plan['files']['semantic_teacher_loss']['sha256']:raise ValueError('Bound semantic loss changed')
    loss_module=types.ModuleType('registered_new_fit_semantic_teacher_loss')
    loss_module.__file__=str(path('semantic_teacher_loss'))
    exec(compile(loss_raw,loss_module.__file__,'exec'),loss_module.__dict__)
    training_teacher_mask=loss_module.training_teacher_mask;teacher_kl_per_row=loss_module.teacher_kl_per_row
    from training.coupled_policy_loss_v10 import make_coupled_policy_mass,policy_binary_cross_entropy
    from training.dropout_readout_v10 import make_head
    from training.prepare_runtime_warm_start_v10 import check_weights,compare_logits
    from training.native_fields_v9 import SIZES
    from roguard.gemma4_review import _score
    validate_files();validate_source_origins()
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
        if len(new_before)!=1540 or sum(r['partition']=='fit' for r in new_before)!=288:
            raise ValueError('Preflight candidate population differs')
        validate_dev_order(before,read_rows(path(language+'_initial_dev_rows')))
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
            if suffix.endswith('features'):validate_feature_norms(values,np)
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
        preflight = dict(format='roguard-new-fit-semantic-mask-study-preflight-1', verification='PASS',
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
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(a.output/'all-language-preflight.json',all_language_preflight)
    report=dict(complete=False,source_sha256=sha(__file__),plan_sha256=a.plan_sha256,languages={},encoder_loaded=False,
                format='roguard-new-fit-semantic-mask-study-result-1',new_baseline_probes_already_exposed=True,
                gate_contract=plan['gate_contract'],scheduled_budget=plan['scheduled_budget'],
                held_out_test_opened=False,qualified=False)
    started=time.monotonic()
    def persist():
        report['seconds']=time.monotonic()-started
        temp=a.output/'report.tmp';write_json(temp,report);temp.replace(a.output/'report.json')
        if object_digest(json.loads((a.output/'report.json').read_text()))!=object_digest(report):raise ValueError('Persisted report differs')
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
        if sorted({new[i]['case_id'] for i in added})!=sorted(expected_ids) or len(expected_ids)!=288:
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
        rec=dict(training_unique_original=len(train),additional_unique=288,additional_occurrences=len(added),arms={})
        report['languages'][lang]=rec
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
        for data_mode in ('new',):
            extra_x,extra_z=(original_x[parents],original_z[parents]) if data_mode=='control' else (new_x[added],new_z[added])
            x=np.concatenate((original_x[train],extra_x));teacher=np.concatenate((original_z[train],extra_z))
            fit_rows=[dict(old[i],fit_origin='original_train') for i in train]+[dict(partition='train',fit_origin='new_fit',fields=new[i]['proposed_fields'],gold=new[i]['proposed_gold']) for i in added]
            fields=np.array([r['fields'] for r in fit_rows],dtype=np.int32);gold=np.array([r['gold'] for r in fit_rows],dtype=np.int32)
            weights=[]
            for j,size in enumerate(SIZES):
                counts=np.bincount(fields[:,j],minlength=size);w=np.minimum(4.,np.sqrt(len(fields)/np.maximum(1,counts)/size));w/=np.mean(w[fields[:,j]])
                weights.append(mx.array(w.astype(np.float32)))
            for distillation in ('mask_wrong_assertion','mask_wrong_new_fit_semantics'):
                arm=data_mode+'-'+distillation;directory=a.output/lang/arm;directory.mkdir(parents=True)
                mask=training_teacher_mask(fit_rows,teacher,distillation)
                if mask.shape!=(len(fit_rows),7) or mask.dtype!=np.bool_ or not mask[:,-1].all():
                    raise ValueError('Semantic mask must retain every legacy binary term')
                np.save(directory/'assertion-keep.npy',mask,allow_pickle=False)
                verify_saved_mask(directory/'assertion-keep.npy',mask,np)
                record=dict(epochs=[],selected=None,mask_false=int((~mask).sum()),mask_shape=list(mask.shape),mask_false_by_term=(~mask).sum(axis=0).tolist(),
                            mask_false_original_by_term=(~mask[:len(train)]).sum(axis=0).tolist(),mask_false_new_fit_by_term=(~mask[len(train):]).sum(axis=0).tolist(),mask_sha256=sha(directory/'assertion-keep.npy'),
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
                    control_reproduction=None
                    if arm=='new-mask_wrong_assertion':
                        control_reference=np.load(path(lang+'_frozen125_epoch_'+str(epoch)+'_dev'),allow_pickle=False)
                        control_reproduction=compare_logits(z,control_reference,spec,limit=1e-3,exact_decisions=True)
                    m,decisions=metrics(z,dev_fields,dev_gold,v9_decisions)
                    m.update(epoch=epoch,updates=epoch*((len(x)+63)//64),presentations=epoch*len(x),introduced_initial75=int(((decisions!=dev_gold)&(initial_decisions==dev_gold)).sum()))
                    if control_reproduction is not None:m['frozen125_control_reproduction']=control_reproduction
                    if epoch:m['training_loss']=float(np.mean(losses))
                    checkpoint=directory/f'epoch-{epoch}.safetensors';zp=directory/f'epoch-{epoch}-dev.npy'
                    tensors={k:np.asarray(v).copy() for k,v in tree_flatten(head.parameters())}
                    save_file(tensors,str(checkpoint));np.save(zp,z,allow_pickle=False)
                    verify_saved_checkpoint(checkpoint,zp,tensors,z,np,load_file)
                    m.update(checkpoint_sha256=sha(checkpoint),dev_logits_sha256=sha(zp),rank=[m['binary_errors'],m['joint_errors'],m['mean_field_ce'],epoch])
                    record['epochs'].append(m)
                    if record['selected'] is None or m['rank']<record['selected']['rank']:record['selected']=dict(m)
                    persist();print(json.dumps(dict(language=lang,arm=arm,**m)),flush=True)
                    if shutil.disk_usage(a.output).free<8*1024**3:raise ValueError('Disk reserve')
                del head,opt,value_grad;gc.collect();mx.clear_cache()
    validate_files()
    validate_source_origins()
    previous=json.loads(path('frozen125_fit_report').read_text())
    for lang,info in report['languages'].items():
        if info['arms']['new-mask_wrong_assertion']['selected']['epoch']!=previous['languages'][lang]['arms']['new-mask_wrong_assertion']['selected']['epoch']:
            raise ValueError('Reproduced125 control selected a different oldDEV epoch')
    frozen={lang:{arm:rec['selected'] for arm,rec in info['arms'].items()} for lang,info in report['languages'].items()}
    verify_all_artifacts(a.output,report,plan,np,load_file)
    write_json(a.output/'selected-before-probes.json',frozen)
    if object_digest(json.loads((a.output/'selected-before-probes.json').read_text()))!=object_digest(frozen):
        raise ValueError('Selected checkpoint freeze readback differs')
    snapshot={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file() and p.name not in ('report.json','report.tmp')}
    validate_files();validate_source_origins()
    if any(sha(a.output/name)!=digest for name,digest in snapshot.items()):raise ValueError('Verified fit artifact changed at completion')
    report['artifacts']=snapshot
    report.update(fitting_complete=True,selection_sha256=sha(a.output/'selected-before-probes.json'),probe_scoring_started=False)
    persist()
    # Deliberately separate probe evaluation into the independently reviewed audit.
    # Evaluation is a separate registered action after all4 selected identities freeze.
    finalize_output(a.output,report,persist,validate_files,validate_source_origins,
        lambda:verify_all_artifacts(a.output,report,plan,np,load_file),snapshot)


if __name__=='__main__':main()
