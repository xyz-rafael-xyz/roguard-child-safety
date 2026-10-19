"""Development-only combiner repair with held-out stress content families."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from training.repaired_three_expert_v9 import load, combine
from training.review_fusion_v9 import read, sha, folds, strata, metrics
from training.boosted_fusion_v9 import CONFIGS, export_model, probability
from training.conservative_fusion_v9 import full_coverage_gap
from training.reserved_metrics_v9 import counts
from training.needs_repair_data_v9 import DEV
from training.train_bare_distress_v9 import texts

ROOT = Path('research/v9-stabilization')
STUDY = ROOT/'stress-aware-fusion-107'


def stress_family(row):
    text = ' '.join(t['text'] for t in row['turns']).casefold()
    families = [phrase for values in DEV.values() for phrase in values if phrase.casefold() in text]
    if len(families) != 1:
        raise ValueError('Stress row must identify exactly one authored content family')
    return hashlib.sha256(families[0].encode()).hexdigest()


def assignments(original, stress):
    values = folds(original) + [int(stress_family(r),16)%5 for r in stress]
    # Exact normalized model inputs cannot cross either a fold or populations.
    seen = {}
    for row,fold in zip(original+stress,values):
        key = texts(row)
        if key in seen and seen[key] != fold:
            raise ValueError('Identical normalized inputs cross folds')
        seen[key] = fold
    return values


def fit(x, y, kinds, config):
    from sklearn.ensemble import GradientBoostingClassifier
    groups = list(zip(kinds,y)); frequencies = Counter(groups)
    if set(frequencies) != {(kind,label) for kind in ('formal','conversational','stress') for label in (0,1)}:
        raise ValueError('Fit lacks a class/population stratum')
    weights = [len(y)/(6*frequencies[g]) for g in groups]
    depth,leaf = CONFIGS[config]
    model = GradientBoostingClassifier(loss='log_loss',n_estimators=100,learning_rate=.05,
        max_depth=depth,min_samples_leaf=leaf,subsample=1.,random_state=1107,n_iter_no_change=None)
    model.fit(x,y,sample_weight=weights)
    result = export_model(model)
    if max(abs(float(a)-probability(result,b)) for a,b in zip(model.predict_proba(x)[:,1],x)) > 1e-12:
        raise ValueError('Portable arithmetic differs')
    return result


def load_development():
    rows,x,y,hashes = load(95,'dev')
    stress_path = ROOT/'installed-repaired-99/stress.jsonl'
    stress = read(stress_path)
    paths = {name: ROOT/'stress-components-106/actual'/(name+'.jsonl')
             for name in ('native','semantic','interpreter')}
    parts = {name:read(path) for name,path in paths.items()}
    ids = [r['case_id'] for r in stress]
    if len(set(ids)) != len(ids) or any([r['case_id'] for r in v] != ids for v in parts.values()):
        raise ValueError('Stress evidence coverage/order differs')
    report_path = ROOT/'stress-components-106/actual/report.json'
    report = json.loads(report_path.read_text())
    if not report['complete'] or report['decision_changes'] or report['maximum_score_error'] > .001:
        raise ValueError('Actual component verification incomplete')
    from training.prepare_instruction_sft_v8 import target
    for row,n,s,i in zip(stress,parts['native'],parts['semantic'],parts['interpreter']):
        gold = target(row)
        if any(v['language'] != 'uk' or v['gold'] != gold for v in (n,s,i)):
            raise ValueError('Stress labels or language differ')
        x.append(combine(n,s,i));y.append(gold)
    hashes.update({'stress':sha(stress_path),'component_report':sha(report_path),
                   **{'stress_'+name:sha(path) for name,path in paths.items()}})
    return rows,stress,x,y,hashes


def group_counts(rows, stress, truth, scores, threshold):
    n = len(rows)
    return dict(metrics(rows,truth[:n],scores[:n],threshold),stress=counts([
        dict(gold=y,predicted=None if s is None else int(s>=threshold)) for y,s in zip(truth[n:],scores[n:])]))


def cutoff(y,scores):
    if any(s is None for s in scores):
        raise ValueError('Abstentions cannot be hidden by a cutoff')
    return full_coverage_gap(y,scores)


def save_predictions(path,rows,truth,scores,threshold):
    path.write_text(''.join(json.dumps(dict(case_id=r['case_id'],language='uk',gold=y,score=s,
        predicted=None if s is None else int(s>=threshold)))+'\n' for r,y,s in zip(rows,truth,scores)))


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('fit','audit'));a=p.parse_args()
    for name,expected in json.loads((STUDY/'source-freeze.json').read_text()).items():
        if sha(name) != expected:raise ValueError('Frozen recipe differs: '+name)
    output=STUDY/'95'
    if a.stage=='fit':
        rows,stress,x,y,hashes=load_development();all_rows=rows+stress
        fold_ids=assignments(rows,stress);kinds=[strata(r) for r in rows]+['stress']*len(stress)
        output.mkdir(exist_ok=False)
        (output/'input-freeze.json').write_text(json.dumps(hashes,indent=2)+'\n')
        trials=[]
        for config in CONFIGS:
            scores=[None]*len(y)
            for fold in range(5):
                train=[i for i in range(len(y)) if fold_ids[i]!=fold and x[i] is not None]
                model=fit([x[i] for i in train],[y[i] for i in train],[kinds[i] for i in train],config)
                for i in range(len(y)):
                    if fold_ids[i]==fold:scores[i]=probability(model,x[i])
            threshold=cutoff(y,scores);groups=group_counts(rows,stress,y,scores,threshold)
            rank=(not all(v['passes'] for v in groups.values()),max(v['false_review_rate'] for v in groups.values()),
                  sum(v['false_review_rate'] for v in groups.values()),CONFIGS[config][0],-CONFIGS[config][1])
            trials.append(dict(config=config,threshold=threshold,groups=groups,rank=rank,scores=scores))
            print('trial',config,json.dumps(groups),flush=True)
        chosen=min(trials,key=lambda t:t['rank'])
        valid=[i for i in range(len(y)) if x[i] is not None]
        model=fit([x[i] for i in valid],[y[i] for i in valid],[kinds[i] for i in valid],chosen['config'])
        fitted=[probability(model,v) for v in x];threshold=cutoff(y+y,chosen['scores']+fitted)
        groups=group_counts(rows,stress,y,fitted,threshold);oof=group_counts(rows,stress,y,chosen['scores'],threshold)
        record=dict(language='uk',parent=95,model=model,threshold=threshold,development_hashes=hashes,
            development={k:v for k,v in groups.items() if k!='stress'},oof={k:v for k,v in oof.items() if k!='stress'},
            stress_development=groups['stress'],stress_oof=oof['stress'],selected_regularization=chosen['config'],
            development_prerequisites_pass=all(v['passes'] for g in (groups,oof) for v in g.values()),
            fold_counts=dict(Counter(fold_ids)),stress_families=len({stress_family(r) for r in stress}),
            trials=[{k:v for k,v in t.items() if k!='scores'} for t in trials],
            selection='Original connected folds plus8whole stresscontentfamilies; equal class/population weight; fullcoverage OOF/refit cutoff.',
            calibration_used_for_selection=False,calibration_diagnosis_informed_experiment=True,
            synthetic_only=True,independent_accuracy_evaluation=False,reserved_probes_used=False,stable_release_ready=False)
        (output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
        save_predictions(output/'dev-predictions.jsonl',rows,y[:len(rows)],fitted[:len(rows)],threshold)
        save_predictions(output/'stress-predictions.jsonl',stress,y[len(rows):],fitted[len(rows):],threshold)
        save_predictions(output/'oof-predictions.jsonl',all_rows,y,chosen['scores'],threshold)
        print('selected',json.dumps({k:record[k] for k in ('development','oof','stress_development','stress_oof','development_prerequisites_pass')}),flush=True)
    else:
        selected=output/'selection.json';record=json.loads(selected.read_text())
        if not record['development_prerequisites_pass']:
            report=dict(selection_sha256=sha(selected),counts_pass=False,calibration_audit_skipped=True,
                reason='Original development or whole-family stress OOF gates fail',reserved_probes_used=False,stable_release_ready=False)
        else:
            if load_development()[-1]!=record['development_hashes']:raise ValueError('Frozen development evidence differs')
            rows,x,y,hashes=load(95,'cal');scores=[probability(record['model'],v) for v in x]
            groups=metrics(rows,y,scores,record['threshold'])
            report=dict(selection_sha256=sha(selected),calibration_hashes=hashes,calibration=groups,
                counts_pass=all(v['passes'] for v in groups.values()),calibration_audit_skipped=False,
                retained_stress_is_development=True,reserved_probes_used=False,stable_release_ready=False)
            save_predictions(output/'cal-predictions.jsonl',rows,y,scores,record['threshold'])
        (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
