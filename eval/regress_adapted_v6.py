"""Regression on consumed v4b test, not a fresh accuracy estimate."""
import argparse,json
from pathlib import Path
from roguard.conversation import ConversationScorer,render
from training.conversations_v4b import generate
from training.adapt_v6 import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve regression records')
    import torch
    torch.set_num_threads(4);rows=generate('test');result={'consumed_regression_only':True,'models':{}}
    for backend in ('semantic','adapted'):
        scorer=ConversationScorer(backend,a.base,a.device);scores=scorer.scores([render(r['turns']) for r in rows],'ro')
        result['models'][backend]=metrics(rows,scores,{l:scorer.thresholds(l)['cutoff'] for l in ('ro','uk')})
        del scorer
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
