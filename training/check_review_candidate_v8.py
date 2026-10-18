"""Fail-closed checks for the direct review model's measured promotion prerequisites.

Passing this study check is necessary, not sufficient, for a stable release:
fresh evaluation and the integrated runtime still require separate verification.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from training.review_data_v8 import input_key
from roguard.evidence import disposition

GROUPS=('ro_formal','ro_conversational','uk_formal','uk_conversational')


def audit_predictions(path, rows, metrics, thresholds):
    """Recompute all counts from complete ID-bound predictions and saved labels."""
    if not path.exists(): return ['saved predictions unavailable']
    try:
        predictions=[json.loads(line) for line in path.read_text().splitlines()]
        by_id={r['case_id']:r for r in rows}
        predicted={r['case_id']:r for r in predictions}
    except (ValueError, KeyError, TypeError): return ['invalid prediction records']
    if len(by_id)!=len(rows) or len(predicted)!=len(predictions) or set(by_id)!=set(predicted):
        return ['prediction IDs are duplicate, missing or unexpected']
    counts={g:dict(positive=0,negative=0,missed=0,false_reviews=0) for g in GROUPS}
    for case_id,row in by_id.items():
        p=predicted[case_id];language=row['language']
        gold=int(disposition({k:{'label':v} for k,v in row['attributes'].items()})[0]!='CONTINUE')
        score=p.get('score');decision=p.get('predicted');threshold=thresholds.get(language)
        if (p.get('language')!=language or type(p.get('gold')) is not int or p['gold']!=gold
                or type(decision) is not int or decision not in (0,1)
                or type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1
                or type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<=threshold<=1
                or decision!=int(score>=threshold)):
            return ['prediction labels, scores or decisions do not match the saved data and threshold']
        kind='conversational' if row.get('source')=='v7_training_frame_projection' else 'formal'
        c=counts[language+'_'+kind];c['positive' if gold else 'negative']+=1
        c['missed']+=int(gold==1 and decision==0);c['false_reviews']+=int(gold==0 and decision==1)
    return [g+': reported counts differ from saved predictions' for g in GROUPS
            if any(metrics.get(g,{}).get(k)!=v for k,v in counts[g].items())]


def check_counts(metrics, groups=GROUPS):
    failures=[]
    for group in groups:
        m=metrics.get(group,{})
        values=[m.get(k) for k in ('positive','negative','missed','false_reviews')]
        if any(type(v) is not int for v in values):
            failures.append(group+': missing or noninteger counts');continue
        positive,negative,missed,false=values
        if positive<=0 or negative<=0 or not 0<=missed<=positive or not 0<=false<=negative:
            failures.append(group+': invalid counts');continue
        # Integer arithmetic avoids rounded rates admitting a failing candidate.
        if missed*100>positive:failures.append(f'{group}: missed {missed}/{positive}; maximum 1%')
        if false*100>negative*5:failures.append(f'{group}: false reviews {false}/{negative}; maximum 5%')
    return failures


def inspect(candidate, language=None):
    candidate=Path(candidate);record=json.loads((candidate/'development.json').read_text())
    failures=[]
    groups=tuple(g for g in GROUPS if language is None or g.startswith(language+'_'))
    artifact_hash=hashlib.sha256((candidate/'review.json.gz').read_bytes()).hexdigest()
    if record.get('artifact_sha256')!=artifact_hash:failures.append('selected artifact hash mismatch or unfinished audit')
    try:
        spec=json.loads(gzip.decompress((candidate/'review.json.gz').read_bytes()))
        thresholds=spec.get('thresholds',{})
        if not isinstance(thresholds,dict): raise ValueError('Invalid thresholds')
    except (OSError, EOFError, ValueError, AttributeError):
        thresholds={};failures.append('invalid selected artifact')
    selected=[e for e in record.get('epochs',[]) if e.get('epoch')==record.get('selected_epoch')]
    if len(selected)!=1:failures.append('missing or ambiguous selected development epoch')
    else:
        failures.extend('development/'+v for v in check_counts(selected[0].get('development',{}),groups))
        if selected[0].get('thresholds')!=thresholds:failures.append('selected epoch and artifact thresholds differ')
    failures.extend('calibration/'+v for v in check_counts(record.get('calibration_audit',{}),groups))
    sets={}
    for split in ('train','dev','cal'):
        rows=[json.loads(l) for l in gzip.decompress((candidate/(split+'.jsonl.gz')).read_bytes()).decode().splitlines()]
        sets[split]={input_key(r) for r in rows}
        if split in ('dev','cal'):
            metrics=(selected[0].get('development',{}) if len(selected)==1 else {}) if split=='dev' else record.get('calibration_audit',{})
            failures.extend(split+'/'+v for v in audit_predictions(candidate/(split+'-predictions.jsonl'),rows,metrics,thresholds))
            for group in GROUPS:
                group_language,kind=group.split('_',1)
                subset=[r for r in rows if r['language']==group_language and ('conversational' if r.get('source')=='v7_training_frame_projection' else 'formal')==kind]
                positives=sum(disposition({k:{'label':v} for k,v in r['attributes'].items()})[0]!='CONTINUE' for r in subset)
                if metrics.get(group,{}).get('positive')!=positives or metrics.get(group,{}).get('negative')!=len(subset)-positives:
                    failures.append(split+'/'+group+': reported denominators differ from saved labels')
        if not rows or len(rows)!=len(sets[split]):failures.append(split+': empty or duplicate normalized inputs')
    for a,b in (('train','dev'),('train','cal'),('dev','cal')):
        if sets[a]&sets[b]:failures.append(a+'/'+b+': overlapping inputs')
    history=candidate/'training-history.jsonl.gz'
    if not history.exists():failures.append('transitive training history unavailable')
    else:
        rows=[json.loads(l) for l in gzip.decompress(history.read_bytes()).decode().splitlines()]
        prior={input_key(r) for r in rows}
        if not sets['train']<=prior:failures.append('training history omits current training inputs')
        for s in ('dev','cal'):
            if prior&sets[s]:failures.append('training history/'+s+': overlapping inputs')
    return {'candidate':str(candidate),'language_scope':language or 'both','artifact_sha256':artifact_hash,'study_prerequisites_pass':not failures,'failures':failures,
            'stable_release_ready':False,'additional_required_checks':['fresh frozen evaluation','integrated runtime accuracy and serious-concern regressions']+(['separately passing other-language candidate'] if language else []),
            'counts_are_synthetic':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--language',choices=('ro','uk'));a=p.parse_args()
    result=inspect(a.candidate,a.language);text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')
    raise SystemExit(0 if result['study_prerequisites_pass'] else 1)

if __name__=='__main__':main()
