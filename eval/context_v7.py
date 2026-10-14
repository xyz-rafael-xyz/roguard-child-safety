"""Fresh frozen context evaluation; do not use the result to tune this candidate."""
import argparse,hashlib,json,time
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
from training.context_v7 import generate,render,digest,LABELS
from training.fit_context_v7 import unique
from roguard.context import ContextModel

def measure(rows,outputs):
    report={}
    for lang in ('ro','uk'):
        pairs=[(r,p) for r,p in zip(rows,outputs) if r['language']==lang];tasks={}
        for task in ('frame','signal'):
            valid=[(r,p['heads'][task]) for r,p in pairs if p['supported']];decided=[(r,p) for r,p in valid if p['label'] is not None]
            tasks[task]={'rows':len(pairs),'supported':len(valid),'argmax_macro_f1':float(f1_score([r[task] for r,p in valid],[p['argmax'] for r,p in valid],average='macro')),
                'decided':len(decided),'decided_errors':sum(r[task]!=p['label'] for r,p in decided),'abstained':len(pairs)-len(decided),
                'set_misses':sum(r[task] not in p['prediction_set'] for r,p in valid)}
            if task=='frame':tasks[task]['labels']=LABELS;tasks[task]['confusion_matrix']=confusion_matrix([r[task] for r,p in valid],[p['argmax'] for r,p in valid],labels=LABELS).tolist()
        groups={}
        for r,p in pairs:groups.setdefault(r['pair_group'],[]).append(p['supported'] and p['heads']['frame']['argmax']==r['frame'])
        report[lang]={'tasks':tasks,'complete_group_count':sum(all(v) for v in groups.values()),'groups':len(groups),'unique_phrase_families':len({r['family'] for r,p in pairs})}
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--skip-v6',action='store_true');p.add_argument('--fresh',action='store_true');p.add_argument('--final3',action='store_true');p.add_argument('--factorial',action='store_true');p.add_argument('--backend',action='append');p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve frozen evaluation')
    freeze=json.loads(Path('research/v7-context/candidate-freeze.json').read_text())
    for backend in (a.backend or ('character','semantic')):
        assert hashlib.sha256(Path('src/roguard/context-'+backend+'.json.gz').read_bytes()).hexdigest()==freeze[backend+'.json.gz']
    from training.context_fresh_v7 import generate as fresh
    from training.context_final_v7 import generate as final3
    from training.context_factorial_final_v7 import generate as factorial
    rows=unique(factorial() if a.factorial else final3() if a.final3 else fresh() if a.fresh else generate('test'));record={'synthetic_only':True,'test_digest':digest(rows),'rows':len(rows),'backends':{}}
    for backend in (a.backend or ('character','semantic')):
        model=ContextModel(backend,a.base,a.device);start=time.monotonic();predictions=model.predict([render(r['turns']) for r in rows if r['language']=='ro'],'ro')+model.predict([render(r['turns']) for r in rows if r['language']=='uk'],'uk')
        record['backends'][backend]={'model_sha256':model.identity,'metrics':measure(rows,predictions),'total_seconds':time.monotonic()-start,'predictions':predictions}
        print(backend,json.dumps(record['backends'][backend]['metrics']),flush=True)
        a.output.write_text(json.dumps(record,indent=2)+'\n');del model
    if a.skip_v6:return
    # v6 has no context frame output. This narrow contrast is comparable only as
    # a current personal signal versus explicitly fictional quotation probe.
    from roguard.conversation import ConversationScorer,render as old_render
    scorer=ConversationScorer('adapted',a.base,a.device)
    subset=[r for r in rows if r['signal']==1 and r['frame'] in ('SELF_CURRENT','QUOTED')]
    comparisons={}
    for lang in ('ro','uk'):
        selected=[r for r in subset if r['language']==lang];scores=scorer.scores([old_render(r['turns']) for r in selected],lang);cut=scorer.thresholds(lang)['cutoff']
        comparisons[lang]={'rows':len(selected),'current_recovered':sum(s>=cut for r,s in zip(selected,scores) if r['frame']=='SELF_CURRENT'),'current_total':sum(r['frame']=='SELF_CURRENT' for r in selected),'quotation_flagged':sum(s>=cut for r,s in zip(selected,scores) if r['frame']=='QUOTED'),'quotation_total':sum(r['frame']=='QUOTED' for r in selected),'cutoff':cut}
    record['v6_narrow_contrast']=comparisons;a.output.write_text(json.dumps(record,indent=2)+'\n');print('v6',comparisons,flush=True)
if __name__=='__main__':main()
