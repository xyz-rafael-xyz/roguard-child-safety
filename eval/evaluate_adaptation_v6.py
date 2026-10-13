"""One fresh test of frozen adapter plus separate calibration; no fitting here."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch
from peft import PeftModel
from roguard.conversation import ConversationScorer,render
from roguard.semantic import SemanticEncoder,head_scores
from roguard.uncertainty import CalibratedDecisions,fingerprint
from training.conversations_v6 import generate,digest
from training.adapt_v6 import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evaluation')
    frozen=json.loads((a.candidate/'freeze.json').read_text())
    for file,sha in frozen.items():
        if hashlib.sha256((a.candidate/file).read_bytes()).hexdigest()!=sha:raise ValueError('Candidate changed after freeze')
    torch.set_num_threads(4);cal=generate('cal');test=generate('test');rows=cal+test;texts=[render(r['turns']) for r in rows]
    baseline=ConversationScorer('semantic',a.base,a.device);base=baseline.scores(texts,'ro')
    encoder=baseline.encoder;encoder.model=PeftModel.from_pretrained(encoder.model,a.candidate/'adapter',is_trainable=False).eval()
    head=json.loads((a.candidate/'head.json').read_text());adapted=head_scores(encoder.encode(texts),head).tolist()
    result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),freeze=frozen,test_digest=digest(test),calibration_digest=digest(cal),synthetic_only=True,independent_language_validation=False,models={})
    for name,scores,cuts in [('v4_semantic',base,{l:baseline.thresholds(l)['cutoff'] for l in ('ro','uk')}),('v6_adapter',adapted,head['cutoffs'])]:
        test_scores=scores[len(cal):];modelid=fingerprint(frozen if name=='v6_adapter' else {'head':baseline.head_sha256,'preprocessing':'surface-v1'})
        entry=dict(metrics=metrics(test,test_scores,cuts),model_id=modelid,rows=[dict(id=r['id'],language=r['language'],family=r['family'],label=r['label'],score=s) for r,s in zip(test,test_scores)],calibration={})
        for lang in ('ro','uk'):
            pairs=[(r,s) for r,s in zip(cal,scores) if r['language']==lang];c=CalibratedDecisions.fit([s for r,s in pairs],[r['label'] for r,s in pairs],model_id=modelid,language=lang)
            outcomes=[(r,c.predict(s,model_id=modelid,language=lang)) for r,s in zip(test,test_scores) if r['language']==lang]
            entry['calibration'][lang]=dict(artifact=c.artifact,test_cases=len(outcomes),covered=sum(r['label'] in p['prediction_set'] for r,p in outcomes),singleton=sum(len(p['prediction_set'])==1 for r,p in outcomes),wrong_singletons=sum(len(p['prediction_set'])==1 and r['label'] not in p['prediction_set'] for r,p in outcomes))
        result['models'][name]=entry
    b=result['models']['v4_semantic']['metrics'];v=result['models']['v6_adapter']['metrics']
    result['promotion_gate_passed']=all(v[l]['recall']>=b[l]['recall'] and v[l]['fpr']<=b[l]['fpr'] for l in ('ro','uk')) and sum(v[l]['balanced_accuracy']-b[l]['balanced_accuracy'] for l in ('ro','uk'))/2>=.02
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('models','freeze')}));print(json.dumps({'baseline':b,'candidate':v}))
if __name__=='__main__':main()
