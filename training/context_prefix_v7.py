"""Train the intermediate prefixes that the incremental runtime actually scores."""
from copy import deepcopy
import unicodedata
from training.context_factorial_v7 import generate as factorial
from training.context_v7 import CONCERNS

def canonical(s):return ''.join(c for c in unicodedata.normalize('NFD',s.lower()) if not unicodedata.combining(c))
def generate(split):
    rows=factorial(split);prefixes=[];seen=set()
    for row in rows:
        if len(row['turns'])<2:continue
        prefix=deepcopy(row);prefix['turns']=row['turns'][:-1];text=prefix['turns'][-1]['text']
        key=(row['language'],text)
        if key in seen:continue
        seen.add(key);prefix['case_id']+='-prefix';prefix['source']='derived_intermediate_prefix'
        concern=canonical(text) in {canonical(t) for t in CONCERNS[row['language']]}
        # A standalone concern is a current first-person candidate. A wrapper
        # without the concern mentions no concern; its scope can still be known.
        prefix['signal']=int(concern)
        if concern:prefix['frame']='SELF_CURRENT'
        elif canonical(text).startswith(('îmi place','imi place','am ','mă bucur','ma bucur','vreau','мені подобається','я ','хочу','радію')):prefix['frame']='ORDINARY'
        prefixes.append(prefix)
    # Explicit oversampling only in training; calibration rows remain unique.
    return rows+prefixes*(4 if split=='train' else 1)
