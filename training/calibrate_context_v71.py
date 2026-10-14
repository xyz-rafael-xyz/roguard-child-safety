"""Class-conditional mention calibration at predeclared 95% synthetic coverage.

Uses only calibration rows. No changes to model weights or frame calibration.
No real-world coverage guarantee under distribution shift.
"""
import argparse,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from roguard.context import ContextModel,render
from training.context_transfer_v71 import generate
from training.fit_context_v7 import unique


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--device',default='cpu');p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 rows=unique(generate('cal'));model=ContextModel('joint',a.base,a.device,a.initial)
 full=model.encoder.encode([render(r['turns']) for r in rows]);focus=model.encoder.encode([render([r['turns'][-1]]) for r in rows]);x=np.concatenate([full,focus],axis=1)
 h=model.spec['heads']['signal'];logits=x@np.asarray(h['weights'])[0]+h['intercept'][0];positive=1/(1+np.exp(-np.clip(logits,-700,700)));probs=np.column_stack([1-positive,positive])
 record={'synthetic_only':True,'initial_sha256':model.identity,'method':'label-conditional split conformal, mention only','alpha':0.05,'cells':{}}
 for lang in ('ro','uk'):
  quantiles=[]
  for label in h['classes']:
   residuals=sorted(1-probs[i,label] for i,r in enumerate(rows) if r['language']==lang and r['signal']==label)
   rank=math.ceil((len(residuals)+1)*.95);q=residuals[rank-1] if rank<=len(residuals) else 1.
   quantiles.append(float(q));record['cells'][f'{lang}/{label}']={'n':len(residuals),'rank':rank,'q':float(q)}
  h['quantiles'][lang]=quantiles
 payload=gzip.compress(json.dumps(model.spec,separators=(',',':')).encode(),mtime=0);(a.output/'joint.json.gz').write_bytes(payload);record['model_sha256']=hashlib.sha256(payload).hexdigest();(a.output/'calibration.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
