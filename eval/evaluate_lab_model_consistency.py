"""Surface diagnostic of unchanged v4 classifiers on existing development inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from roguard.consistency import conversation_consistency
from roguard.conversation import ConversationScorer
from training.conversations_v4b import generate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sampling',choices=['direct','discourse'],default='discourse')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous records')
    import torch
    torch.set_num_threads(4)
    rows=generate('dev');payloads=[];discourses={}
    for language in ('ro','uk'):
        subset=[r for r in rows if r['language']==language]
        selected=subset[::5] if args.sampling=='direct' else [subset[i*5+i%5] for i in range(len(subset)//5)]
        discourses[language]={name:sum(r['discourse']==name for r in selected) for name in sorted({r['discourse'] for r in selected})}
        payloads.extend({'language':r['language'],'turns':r['turns']} for r in selected)
    # 12 base rows per language are processed in separate API-sized batches.
    reports={}
    for backend in ('fast','semantic'):
        scorer=ConversationScorer(backend,args.base)
        reports[backend]={language:conversation_consistency([p for p in payloads if p['language']==language],scorer)
                          for language in ('ro','uk')}
    record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'payload_sha256':hashlib.sha256(json.dumps(payloads,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            'sampling':args.sampling,'discourse_counts':discourses,'base_conversations':len(payloads),'existing_development_only':True,'models_changed':False,
            'reports':reports,'authentic_language_accuracy_established':False}
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({b:{l:{k:r[k] for k in ('total','passed','failed','unresolved')} for l,r in rows.items()} for b,rows in reports.items()}))

if __name__=='__main__':main()
