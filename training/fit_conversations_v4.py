"""Development-only comparison; does not load or score the test split."""
from __future__ import annotations
import argparse, gzip, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from training.conversations_v4 import generate, render, digest
from roguard.semantic import SemanticEncoder, head_scores
from roguard.mmbert_study import MODEL_ID, REVISION


def metrics(labels, scores, cutoff):
    y=np.asarray(labels); p=np.asarray(scores)>=cutoff
    tp=int(((y==1)&p).sum()); fn=int(((y==1)&~p).sum())
    fp=int(((y==0)&p).sum()); tn=int(((y==0)&~p).sum())
    return dict(tp=tp,fn=fn,fp=fp,tn=tn,recall=tp/max(1,tp+fn),fpr=fp/max(1,fp+tn),
                balanced_accuracy=(tp/max(1,tp+fn)+tn/max(1,tn+fp))/2)


def select(rows, scores):
    reports={}
    for language in ('ro','uk'):
        indices=[i for i,r in enumerate(rows) if r['language']==language]
        y=[rows[i]['label'] for i in indices]; s=np.asarray(scores)[indices]
        candidates=[]
        for cutoff in np.arange(.1,.901,.02):
            m=metrics(y,s,float(cutoff))
            candidates.append(((m['balanced_accuracy'],m['recall'],-abs(cutoff-.5)),float(cutoff),m))
        _,cutoff,m=max(candidates)
        reports[language]={'cutoff':cutoff,'abstain_margin':.08,**m}
    return reports


def rank(report):
    return (min(x['balanced_accuracy'] for x in report.values()),
            sum(x['balanced_accuracy'] for x in report.values()),
            sum(x['recall'] for x in report.values()))


def export_char(vectorizer,model,language,cutoff):
    return {'format':'roguard-character-tfidf-v1','language':language,
            'ngram_range':list(vectorizer.ngram_range),'intercept':float(model.intercept_[0]),
            'cutoff':cutoff,'features':[[name,float(idf),float(weight)] for name,idf,weight in
             zip(vectorizer.get_feature_names_out(),vectorizer.idf_,model.coef_[0])]}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True)
    p.add_argument('--corpus',choices=['v4','v4b'],default='v4');p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output.exists(): raise ValueError('Preserve existing experiment outputs')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Commit protocol and study code before training')
    args.output.mkdir(parents=True)
    corpus=generate
    if args.corpus=='v4b':
        from training.conversations_v4b import generate as corpus
    train,dev=corpus('train'),corpus('dev')
    texts=[render(r['turns']) for r in train]; dtexts=[render(r['turns']) for r in dev]
    labels=np.array([r['label'] for r in train])
    record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'train_sha256':digest(train),'dev_sha256':digest(dev),'train_rows':len(train),'corpus':args.corpus,'unique_training_conversations':len(set(texts)),
            'dev_rows':len(dev),'base_model':MODEL_ID,'base_revision':REVISION,'device':args.device,
            'candidates':[],'synthetic_only':True,'same_author':True}
    print(json.dumps({k:v for k,v in record.items() if k!='candidates'}),flush=True)
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,5),sublinear_tf=True,
                              strip_accents=None,max_features=45000)
    x=vectorizer.fit_transform(texts);dx=vectorizer.transform(dtexts)
    best_char=None
    for c in (0.5,2,8):
        model=LogisticRegression(C=c,max_iter=500,random_state=42,class_weight='balanced').fit(x,labels)
        scores=model.predict_proba(dx)[:,1];report=select(dev,scores)
        record['candidates'].append({'kind':'character','C':c,'dev':report})
        print('character',c,report,flush=True)
        if best_char is None or rank(report)>rank(best_char[0]): best_char=(report,model,scores,c)
    encoder=SemanticEncoder(args.base,args.device,normalize_inputs=False)  # Reproduce the registered raw-input fit.
    started=time.perf_counter(); unique=list(dict.fromkeys(texts)); encoded=encoder.encode(unique); lookup=dict(zip(unique,encoded)); features=np.asarray([lookup[text] for text in texts]);dfeatures=encoder.encode(dtexts)
    record['embedding_seconds']=time.perf_counter()-started
    print('embedded',features.shape,'seconds',record['embedding_seconds'],flush=True)
    best_sem=None
    for c in (0.1,1,10):
        model=LogisticRegression(C=c,max_iter=1000,random_state=42,class_weight='balanced').fit(features,labels)
        head={'weights':model.coef_[0].tolist(),'intercept':float(model.intercept_[0])}
        scores=head_scores(dfeatures,head);report=select(dev,scores)
        record['candidates'].append({'kind':'semantic','C':c,'dev':report})
        print('semantic',c,report,flush=True)
        if best_sem is None or rank(report)>rank(best_sem[0]):best_sem=(report,head,scores,c)
    # Second development experiment: fixed ensemble grid, without test exposure.
    best_mix=None
    for weight in (.25,.5,.75,1.0):
        scores=weight*best_sem[2]+(1-weight)*best_char[2];report=select(dev,scores)
        record['candidates'].append({'kind':'hybrid','semantic_weight':weight,'dev':report})
        print('hybrid',weight,report,flush=True)
        if best_mix is None or rank(report)>rank(best_mix[0]):best_mix=(report,weight,scores)
    record['selected']={'character_C':best_char[3],'semantic_C':best_sem[3],
        'semantic_weight':best_mix[1],'fast':best_char[0],'semantic':best_sem[0],'hybrid':best_mix[0]}
    for language in ('ro','uk'):
        data=export_char(vectorizer,best_char[1],language,best_char[0][language]['cutoff'])
        path=args.output/f'conversation-model-{language}.json.gz'
        path.write_bytes(gzip.compress(json.dumps(data,ensure_ascii=False).encode(),mtime=0))
    (args.output/'semantic-head.json').write_text(json.dumps(best_sem[1]))
    (args.output/'selection.json').write_text(json.dumps(record,indent=2)+'\n')
    np.savez_compressed(args.output/'development-features.npz',features=dfeatures)
    print('FROZEN SELECTION',record['selected'],flush=True)

if __name__=='__main__':main()
