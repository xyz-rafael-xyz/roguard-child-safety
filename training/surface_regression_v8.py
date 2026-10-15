"""Canonical Unicode and whitespace invariance on existing synthetic regressions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import types
import unicodedata
from roguard.context import ContextModel, render
from roguard.evidence import EvidenceModel
from training.evidence_v8 import load


def old_features(self, text):
    if isinstance(self, EvidenceModel):
        text=re.sub(r'\s\s+',' ',text.lower())
        counts=Counter(text[i:i+n] for n in range(2,6) for i in range(len(text)-n+1))
    else:
        counts=Counter(text.lower()[i:i+n] for n in range(2,6) for i in range(len(text)-n+1))
    values={self.features[s][0]:(1+math.log(n))*self.features[s][1] for s,n in counts.items() if s in self.features}
    norm=math.sqrt(sum(v*v for v in values.values()))
    return {i:v/norm for i,v in values.items()} if norm else {}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Refusing overwrite')
    rows=load(Path('research/v8-evidence/data'),'test')[::20]
    report={'partition':'previously inspected v8 test regression','synthetic_only':True,'cases':len(rows),'models':{}}
    for name,model in [('evidence',EvidenceModel()),('context',ContextModel('character'))]:
        current=model._features;stats={};canonical_changes=0
        for row in rows:
            language=row['language']
            text=row['turns'][row['focus']]['text'] if name=='evidence' else render(row['turns'][:row['focus']+1])
            model._features=current;reference=model.predict([text],language)
            model._features=types.MethodType(old_features,model);old_reference=model.predict([text],language)
            canonical_changes+=reference!=old_reference
            for surface,changed in [('NFD',unicodedata.normalize('NFD',text)),('repeated_spaces',text.replace(' ','  '))]:
                key=language+'_'+surface;count=stats.setdefault(key,{'cases':0,'before_mismatches':0,'after_mismatches':0})
                count['cases']+=1
                model._features=types.MethodType(old_features,model);count['before_mismatches']+=model.predict([changed],language)!=old_reference
                model._features=current;count['after_mismatches']+=model.predict([changed],language)!=reference
        report['models'][name]={'canonical_output_changes':canonical_changes,'groups':stats}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
