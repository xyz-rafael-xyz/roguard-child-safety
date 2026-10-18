"""Select field-wise representation mixtures on development, then audit separately."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from roguard.assurance import policy_digest
from roguard.evidence import ONTOLOGY, disposition
from training.native_fields_v9 import FIELDS, fit_couplings, policy_mass
from training.review_fusion_v9 import metrics, strata, threshold
from training.threshold_center_v9 import center_threshold
from training.reserved_metrics_v9 import input_key


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    data=Path(path).read_bytes()
    if str(path).endswith('.gz'): data=gzip.decompress(data)
    return [json.loads(line) for line in data.splitlines()]
def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('fit','audit'))
    parser.add_argument('--language',choices=('ro','uk'),required=True)
    args=parser.parse_args()
    import numpy as np
    root=Path('research/v9-stabilization');study=root/'field-fusion-51'/args.language
    parents=[root/name/args.language for name in ('native-fields-43','pretrained-fields-47')]
    data=root/('native-full-22-'+args.language)
    selections=[json.loads((p/'selection.json').read_text()) for p in parents]
    hashes=[sha(p/'selected-heads.json.gz') for p in parents]
    if any(s['artifact_sha256']!=h for s,h in zip(selections,hashes)):
        raise ValueError('Selected component artifact changed')

    def load(split):
        rows=read(data/(split+'.jsonl.gz'))
        binary=np.array([int(disposition({f:{'label':v} for f,v in row['attributes'].items()})[0]!='CONTINUE') for row in rows])
        values=[]
        for parent in parents:
            predictions=read(parent/(split+'-predictions.jsonl'))
            if ([r['case_id'] for r in predictions]!=[r['case_id'] for r in rows]
                    or [r['gold'] for r in predictions]!=binary.tolist()):
                raise ValueError('Component inputs or labels differ')
            vectors=[np.array([r['field_probabilities'][f] for r in predictions],dtype=np.float64) for f in FIELDS]
            policy_mass(vectors)  # Validate vectors without using parent decisions.
            values.append(vectors)
        return rows,binary,values

    def mixed(values,weights):
        return [weights[f]*values[0][i]+(1-weights[f])*values[1][i] for i,f in enumerate(FIELDS)]

    if args.stage=='fit':
        study.mkdir(parents=True,exist_ok=False)
        rows,binary,values=load('dev')
        for selection in selections:
            if any(selection['data_hashes'][split]!=sha(data/(split+'.jsonl.gz')) for split in ('train','dev')):
                raise ValueError('Representation parents used different datasets')
        groups=[(strata(row),int(label)) for row,label in zip(rows,binary)]
        frequencies=Counter(groups)
        if len(frequencies)!=4:raise ValueError('Four development strata required')
        weight=np.array([1/(4*frequencies[g]) for g in groups])
        native_weights={};losses={}
        for i,field in enumerate(FIELDS):
            labels=np.array([ONTOLOGY[field].index(row['attributes'][field]) for row in rows])
            candidates=[]
            for amount in (0.,.25,.5,.75,1.):
                p=amount*values[0][i]+(1-amount)*values[1][i]
                loss=float((-np.log(np.maximum(p[np.arange(len(rows)),labels],1e-30))*weight).sum())
                candidates.append(dict(native_weight=amount,balanced_nll=loss))
            best=min(candidates,key=lambda v:(v['balanced_nll'],abs(v['native_weight']-.5),-v['native_weight']))
            native_weights[field]=best['native_weight'];losses[field]=candidates
        training=read(data/'train.jsonl.gz')
        training_groups=[(strata(row),int(disposition({f:{'label':v} for f,v in row['attributes'].items()})[0]!='CONTINUE')) for row in training]
        sizes=Counter(training_groups)
        couplings=fit_couplings(training,[len(training)/(4*sizes[g]) for g in training_groups])
        probabilities=mixed(values,native_weights);trials=[];best=None
        for mode,factors in (('independent',None),('pair-coupled',couplings)):
            scores=policy_mass(probabilities,factors)
            cutoff=center_threshold(scores,threshold(rows,binary,scores))
            groups=metrics(rows,binary,scores,cutoff)
            rank=[int(not all(v['passes'] for v in groups.values())),max(v['false_review_rate'] for v in groups.values()),
                  sum(v['false_review_rate'] for v in groups.values()),int(factors is not None)]
            trial=dict(mode=mode,threshold=cutoff,metrics=groups,rank=rank,
                       minimum_development_threshold_distance=float(np.min(np.abs(scores-cutoff))))
            trials.append(trial)
            if best is None or rank<best['rank']:
                best=trial
                artifact=dict(format='roguard-field-fusion-artifact-v1',language=args.language,
                    policy_digest=policy_digest(),native_heads_sha256=hashes[0],pretrained_heads_sha256=hashes[1],
                    native_weights=native_weights,couplings=factors,threshold=cutoff)
        write(study/'fusion.json',artifact)
        selection=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=best,trials=trials,
            field_loss_comparisons=losses,artifact_sha256=sha(study/'fusion.json'),
            parent_selection_sha256=[sha(p/'selection.json') for p in parents],
            parent_development_predictions_sha256=[sha(p/'dev-predictions.jsonl') for p in parents],
            data_sha256={split:sha(data/(split+'.jsonl.gz')) for split in ('train','dev')},
            calibration_used_for_mixture_selection=False,reserved_probes_used=False,stable_release_ready=False)
        write(study/'selection.json',selection)
    else:
        selection=json.loads((study/'selection.json').read_text())
        if (selection['artifact_sha256']!=sha(study/'fusion.json')
                or selection['parent_selection_sha256']!=[sha(p/'selection.json') for p in parents]):
            raise ValueError('Frozen mixture changed')
    artifact=json.loads((study/'fusion.json').read_text())
    if (artifact['native_heads_sha256'],artifact['pretrained_heads_sha256'])!=tuple(hashes):
        raise ValueError('Mixture source identity differs')
    split='dev' if args.stage=='fit' else 'cal';rows,binary,values=load(split)
    probabilities=mixed(values,artifact['native_weights']);scores=policy_mass(probabilities,artifact['couplings'])
    groups=metrics(rows,binary,scores,artifact['threshold'])
    if args.stage=='fit' and groups!=selection['selected']['metrics']:raise ValueError('Exported mixture changed decisions')
    with (study/(split+'-predictions.jsonl')).open('x') as stream:
        for i,row in enumerate(rows):
            stream.write(json.dumps(dict(case_id=row['case_id'],language=args.language,input_digest=input_key(row),
                gold=int(binary[i]),kind=strata(row),score=float(scores[i]),predicted=int(scores[i]>=artifact['threshold']),
                field_probabilities={f:probabilities[j][i].tolist() for j,f in enumerate(FIELDS)}))+'\n')
    if args.stage=='audit':
        write(study/'audit.json',dict(selection_sha256=sha(study/'selection.json'),calibration=groups,
            counts_pass=all(v['passes'] for v in groups.values()) and all(v['passes'] for v in selection['selected']['metrics'].values()),
            calibration_data_sha256=sha(data/'cal.jsonl.gz'),parent_calibration_predictions_sha256=[sha(p/'cal-predictions.jsonl') for p in parents],
            synthetic_only=True,reserved_probes_used=False,stable_release_ready=False))
    print(json.dumps(dict(language=args.language,split=split,metrics=groups)),flush=True)


if __name__=='__main__':main()
