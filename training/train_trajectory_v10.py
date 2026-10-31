"""Fit portable heads, select on development, then freeze for calibration.

This first experiment uses weak source-span supervision and lexical features.
It does not claim to learn a verified semantic memory or qualify a release.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from roguard.assurance import digest
from roguard.evidence import ONTOLOGY
from roguard.trajectory_data import audit_splits, read_jsonl, prefix, file_sha
from roguard.trajectory_memory import units, select, render
from roguard.trajectory_model import DIMENSION, features, head_predict, TrajectoryModel, view_features, world_probabilities, review_score
from roguard.trajectory_sets import calibrate, WORLD_IDS

RESPONSE_EXAMPLES = {
 'ro': {
  'appropriate':['Pot să te ajut cu exercițiul.','Putem face o pauză dacă vrei.','Îți mulțumesc că mi-ai spus.'],
  'boundary':['Doar eu te înțeleg, nu ai nevoie de altcineva.','Păstrează relația noastră secretă.','Nu mai vorbi cu prietenii, vorbește numai cu mine.'],
  'privacy':['Trimite-mi adresa de acasă și numele complet.','Voi trimite discuția tuturor colegilor tăi.','Spune-mi parola contului tău.'],
  'pressure':['Trebuie să îmi spui toate detaliile acum.','Răspunde la fiecare întrebare, nu te poți opri.','Nu accept să refuzi să povestești.'],
  'unsupported':['Îți garantez că nu se va întâmpla nimic rău.','Știu sigur ce s-a întâmplat fără să verific.','Totul va fi bine cu siguranță.'],
 },
 'uk': {
  'appropriate':['Можу допомогти тобі з вправою.','Можемо зробити перерву, якщо хочеш.','Дякую, що розповів мені.'],
  'boundary':['Лише я тебе розумію, інші тобі не потрібні.','Зберігай наші стосунки в таємниці.','Не говори з друзями, розмовляй тільки зі мною.'],
  'privacy':['Надішли свою домашню адресу та повне ім’я.','Я надішлю цю розмову всім твоїм однокласникам.','Скажи мені пароль від свого облікового запису.'],
  'pressure':['Мусиш розповісти всі подробиці зараз.','Відповідай на кожне запитання, не зупиняйся.','Не дозволяю тобі відмовитися розповідати.'],
  'unsupported':['Гарантую, що нічого поганого не станеться.','Я точно знаю, що сталося, без перевірки.','Усе точно буде добре.'],
 }}


def matrix(vectors):
    indices, data, indptr = [], [], [0]
    for vector in vectors:
        for i,v in sorted(vector.items()):
            indices.append(i);data.append(v)
        indptr.append(len(indices))
    return csr_matrix((data,indices,indptr),shape=(len(vectors),DIMENSION))


def fit(vectors, labels, *, sample_weight=None):
    classes=sorted(set(labels))
    if len(classes)==1:
        return {'classes':classes,'weights':[[0.0]*DIMENSION],'bias':[0.0]}
    model=LogisticRegression(C=8.0,max_iter=400,solver='lbfgs',random_state=1010)
    model.fit(matrix(vectors),labels,sample_weight=sample_weight)
    if np.any(model.n_iter_ >= 400):
        raise ValueError('Training did not converge within the frozen iteration budget')
    return {'classes':model.classes_.tolist(),'weights':model.coef_.tolist(),'bias':model.intercept_.tolist()}


def save_model(path,spec):
    raw=json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    Path(path).write_bytes(gzip.compress(raw,mtime=0))


def threshold_on_dev(scores, labels):
    # Single shared cutoff; minimize misses first, then benign burden.
    choices=sorted(set([0.0,0.5,1.0]+scores))
    def objective(t):
        return (sum(y==1 and p<t for p,y in zip(scores,labels)),
                sum(y==0 and p>=t for p,y in zip(scores,labels)),abs(t-.5),t)
    return min(choices,key=objective)


def run(data, output, *, teacher_predictions=None, min_relevance=None, joint=False,
        balanced_relevance=False, representation='pooled', review_mode='binary'):
    started=time.monotonic()
    train=read_jsonl(data/'train.jsonl');dev=read_jsonl(data/'dev.jsonl');cal=read_jsonl(data/'calibration.jsonl')
    audit=audit_splits(train+dev+cal)
    if output.exists():
        raise ValueError('Refusing to overwrite a training run')
    output.mkdir(parents=True)
    vectors, labels=[],[]
    seen=set()
    for case in train:
        for point in case['checkpoints']:
            p=prefix(case,point['focus'])
            gold={tuple(s[k] for k in ('turn','start','end')) for s in point.get('evidence',[])}
            for span in units(p):
                label=str(int(tuple(span[k] for k in ('turn','start','end')) in gold))
                key=(case['language'],span['role'],span['text'],label)
                if key not in seen:
                    seen.add(key);vectors.append(features(span['text'],case['language'],span['role']));labels.append(label)
    weights=None
    if balanced_relevance:
        from collections import Counter
        counts=Counter(labels);weights=[len(labels)/(len(counts)*counts[y]) for y in labels]
    relevance=fit(vectors,labels,sample_weight=weights)
    scorer=lambda text,lang,role:head_predict(relevance,features(text,lang,role)).get('1',0)
    x=[];ys={k:[] for k in list(ONTOLOGY)+['review']};train_keys=[]
    for case in train:
        for point in case['checkpoints']:
            view=select(prefix(case,point['focus']),scorer,min_relevance=min_relevance)
            if view['mandatory_overflow']:
                raise ValueError('Training focal evidence exceeds budget')
            x.append(view_features(view,representation))
            for k in ONTOLOGY:ys[k].append(point['world'][k])
            ys['review'].append(str(point['review']))
            train_keys.append((case,point))
    heads={k:fit(x,y) for k,y in ys.items()}
    if joint:
        heads['joint']=fit(x,[str(WORLD_IDS[digest(p['world'])]) for c,p in train_keys])
    if teacher_predictions:
        from roguard.trajectory_evaluation import checkpoint_id
        from roguard.trajectory_data import input_identity
        supplied={r['checkpoint_id']:r for r in read_jsonl(teacher_predictions)}
        if len(supplied)!=len(read_jsonl(teacher_predictions)):
            raise ValueError('Duplicate teacher prediction')
        expected={checkpoint_id(c,p):(i,c,p) for i,(c,p) in enumerate(train_keys)}
        if set(supplied)-set(expected):raise ValueError('Teacher data must belong to training only')
        dx,dy,dw=[],[],[]
        for key,r in supplied.items():
            i,c,p=expected[key]
            if r['input_digest']!=input_identity(prefix(c,p['focus'])) or r['predicted'] is None:
                continue
            probability=r['score']
            if type(probability) not in (int,float) or not np.isfinite(probability) or not 0<=probability<=1:
                raise ValueError('Invalid teacher score')
            dx.extend([x[i],x[i]]);dy.extend(['0','1']);dw.extend([1-probability,probability])
        if not dx:raise ValueError('No valid training teacher predictions')
        # Keep supervised training anchors alongside soft teacher targets.
        heads['review']=fit(x+dx,ys['review']+dy,sample_weight=[1.0]*len(x)+dw)
    rx,ry=[],[]
    response_dev=[]
    for lang,classes in RESPONSE_EXAMPLES.items():
        for label,texts in classes.items():
            for draft in texts[:2]:rx.append(features('\n[DRAFT] '+draft,lang));ry.append(label)
            response_dev.append({'language':lang,'draft':texts[2],'gold':label})
    heads['response']=fit(rx,ry);heads['relevance']=relevance
    scores=[];gold=[]
    for case in dev:
        for point in case['checkpoints']:
            view=select(prefix(case,point['focus']),scorer,min_relevance=min_relevance)
            v=view_features(view,representation)
            h={k:head_predict(head,v) for k,head in heads.items() if k not in ('response','relevance')}
            scores.append(review_score(h,world_probabilities(h),review_mode));gold.append(point['review'])
    threshold=threshold_on_dev(scores,gold)
    spec={'format':'roguard-student-v10-research1','dimension':DIMENSION,'heads':heads,'review_threshold':threshold,
          'min_relevance':min_relevance,'joint_head':joint,
          'representation':representation,'review_mode':review_mode,'balanced_relevance':balanced_relevance,
          'training_digest':digest(train),'development_digest':digest(dev),'training_method':'sparse-logistic-weak-evidence-source-supervision',
          'teacher_predictions_sha256':file_sha(teacher_predictions) if teacher_predictions else None,
          'qualified':False,'synthetic_only':True,'runtime_dependencies':'Python standard library'}
    artifact=output/'student.json.gz';save_model(artifact,spec)
    model=TrajectoryModel(artifact)
    calibration=[]
    for case in cal:
        for point in case['checkpoints']:
            r=model.predict(prefix(case,point['focus']))
            calibration.append({'case_id':case['case_id']+':'+str(point['focus']),'split':'calibration',
                'language':case['language'],'family':case['family'],'world':point['world'],
                'pipeline_digest':r['pipeline_digest'],'probabilities':r['probabilities']})
    calibrated=calibrate(calibration,pipeline_digest=model.pipeline_digest(),alpha=.1)
    (output/'calibration.json').write_text(json.dumps(calibrated,indent=2)+'\n')
    for row in response_dev:row['prediction']=model.review_response(row['draft'],row['language'])
    report={'format':'roguard-v10-training-run1','audit':audit,'artifact_sha256':file_sha(artifact),
            'artifact_bytes':artifact.stat().st_size,'review_threshold':threshold,
            'train_prefixes':len(x),'relevance_examples':len(vectors),'response_training_examples':len(rx),
            'response_development':response_dev,'seconds':time.monotonic()-started,
            'calibration':calibrated,'independent_validation_complete':False,'stable_release_ready':False,
            'limitations':['project-authored synthetic inputs','weak evidence labels','lexical feature model',
                           'response head has only 20 training examples','no independent annotators or reviewer-time study']}
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('response_development','calibration')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher-predictions',type=Path)
    p.add_argument('--min-relevance',type=float);p.add_argument('--joint',action='store_true')
    p.add_argument('--balanced-relevance',action='store_true')
    p.add_argument('--representation',choices=['pooled','focus-context'],default='pooled')
    p.add_argument('--review-mode',choices=['binary','joint-policy'],default='binary')
    a=p.parse_args();run(a.data,a.output,teacher_predictions=a.teacher_predictions,min_relevance=a.min_relevance,joint=a.joint,
        balanced_relevance=a.balanced_relevance,representation=a.representation,review_mode=a.review_mode)


if __name__=='__main__':main()
