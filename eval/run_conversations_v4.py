"""Score a frozen selection once; preserve results and dependency structure."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from training.conversations_v4 import generate,render,digest
from training.fit_conversations_v4 import metrics
from roguard.conversation import ConversationScorer,configuration
from roguard.message_model import SyntheticMessageModel
from roguard.semantic import head_scores


def interval(success,total):
    if not total:return None
    z=1.959964;p=success/total;den=1+z*z/total
    mid=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [round(mid-half,4),round(mid+half,4)]


def summarize(rows,scores,settings):
    result={}
    for language in ('ro','uk'):
        indices=[i for i,r in enumerate(rows) if r['language']==language]
        group=[rows[i] for i in indices];s=np.asarray(scores)[indices];y=np.array([r['label'] for r in group])
        cutoff=settings[language]['cutoff'];margin=settings[language].get('abstain_margin',0)
        m=metrics(y,s,cutoff);pred=s>=cutoff;unknown=np.abs(s-cutoff)<=margin
        families=defaultdict(list)
        for row,correct in zip(group,pred==y):families[row['family']].append(bool(correct))
        decided=~unknown
        per={}
        for d in sorted({r['discourse'] for r in group}):
            ii=[i for i,r in enumerate(group) if r['discourse']==d]
            per[d]={'total':len(ii),'errors':int((pred[ii]!=y[ii]).sum()),'abstentions':int(unknown[ii].sum())}
        result[language]={**m,'precision':m['tp']/max(1,m['tp']+m['fp']),
            'false_alerts_per_1000_negatives':1000*m['fpr'],
            'recall_wilson_descriptive':interval(m['tp'],m['tp']+m['fn']),
            'fpr_wilson_descriptive':interval(m['fp'],m['fp']+m['tn']),
            'complete_families':sum(all(v) for v in families.values()),'families':len(families),
            'setting_groups':len({r['setting_group'] for r in group}),
            'need_groups':len({r['need_group'] for r in group}),
            'coverage':float(decided.mean()),'abstentions':int(unknown.sum()),
            'decided_misses':int(((y==1)&~pred&decided).sum()),
            'decided_false_alerts':int(((y==0)&pred&decided).sum()),
            'overall_review_or_unresolved':int((pred|unknown).sum()),
            'per_discourse':per}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output.exists():raise ValueError('Evaluation output exists; preserve prior results')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Freeze selection in Git first')
    selection=json.loads(Path('research/v4-conversation-selection/selection.json').read_text())
    for name,expected in selection['artifact_sha256'].items():
        assert hashlib.sha256((Path('src/roguard')/name).read_bytes()).hexdigest()==expected
    rows=generate('test');config=configuration();fast=ConversationScorer();semantic=ConversationScorer('semantic',args.base,args.device)
    started=time.perf_counter();cs=[];ss=[];old=[]
    for language in ('ro','uk'):
        subset=[r for r in rows if r['language']==language]
        texts=[render(r['turns']) for r in subset]
        cs.extend(fast.scores(texts,language))
        ss.extend(head_scores(semantic.encoder.encode(texts),semantic.head).tolist())
        baseline=SyntheticMessageModel.packaged(language)
        old.extend(baseline.score(next(t['text'] for t in reversed(r['turns']) if t['role']=='minor')) for r in subset)
    mixed=[config['semantic_weight']*s+(1-config['semantic_weight'])*c for s,c in zip(ss,cs)]
    old_config={lang:{'cutoff':SyntheticMessageModel.packaged(lang).cutoff} for lang in ('ro','uk')}
    reports={name:summarize(rows,score,setting) for name,score,setting in [
        ('v3_last_message',old,old_config),('fast',cs,config['fast']),('semantic',ss,config['semantic']),('hybrid',mixed,config['hybrid'])]}
    predictions=[{'id':r['id'],'family':r['family'],'language':r['language'],'label':r['label'],
                  'scores':dict(v3=old[i],fast=cs[i],semantic=ss[i],hybrid=mixed[i])} for i,r in enumerate(rows)]
    record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'test_sha256':digest(rows),'test_rows':len(rows),'seconds':time.perf_counter()-started,
        'reports':reports,'synthetic_only':True,'independent_author_validation':False,
        'row_intervals_ignore_shared_templates':True,'predictions':predictions}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
