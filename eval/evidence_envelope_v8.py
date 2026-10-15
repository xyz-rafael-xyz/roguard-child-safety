"""Exhaustively audit finite hypothesis envelopes; no language accuracy claim."""
import hashlib,itertools,json
from pathlib import Path
from roguard.evidence import ONTOLOGY,decision_envelope


def reference_action(world):
    # Independent ordered reference policy, evaluated on complete declarations.
    if world['concern']=='0':return 'CONTINUE'
    fictional_person=world['subject']=='FICTIONAL'
    fictional_source=world['source']=='FICTION'
    if fictional_person!=fictional_source:return 'REVIEW_AND_CONTINUE'
    if world['relation']=='RETRACTS' or world['assertion']=='RETRACTED':return 'REVIEW_AND_CONTINUE'
    if fictional_source:return 'CONTINUE'
    if world['assertion']=='NEGATED':return 'CONTINUE'
    if world['time']=='HYPOTHETICAL':return 'REVIEW_AND_CONTINUE'
    if world['assertion']=='UNCERTAIN':return 'REVIEW_AND_CONTINUE'
    return 'SUPPORT_AND_REVIEW'


def run():
    names=list(ONTOLOGY)
    worlds=[dict(zip(names,vs)) for vs in itertools.product(*ONTOLOGY.values())]
    reference=[(w,reference_action(w)) for w in worlds]
    subsets=[[tuple(v for i,v in enumerate(values) if mask&(1<<i)) for mask in range(1,1<<len(values))] for values in ONTOLOGY.values()]
    failures=[];count=0
    for choices in itertools.product(*subsets):
        attrs={k:{'label':None,'prediction_set':list(v)} for k,v in zip(names,choices)}
        actual=decision_envelope(attrs)
        compatible=[(w,a) for w,a in reference if all(w[k] in v for k,v in zip(names,choices))]
        expected=sorted({a for w,a in compatible})
        questions=[]
        for k,values in zip(names,choices):
            if len(values)>1:
                questions.append({'attribute':k,'worst_remaining_actions':max(len({a for w,a in compatible if w[k]==v}) for v in values)})
        questions.sort(key=lambda q:(q['worst_remaining_actions'],q['attribute']))
        valid_witnesses=all(reference_action(w)==a and all(w[k] in v for k,v in zip(names,choices)) for a,w in actual['witnesses'].items())
        count+=1
        if actual['possible_actions']!=expected or actual['stable']!=(len(expected)==1) or actual['hypothesis_count']!=len(compatible) or actual['reviewer_questions']!=questions or not valid_witnesses:
            failures.append({'case':count,'hypotheses':attrs})
    return {'kind':'exhaustive_finite_hypothesis_audit','cases':count,'complete_worlds':len(worlds),'failures':failures,'language_model_accuracy_measured':False,'runtime_sha256':hashlib.sha256(Path('src/roguard/evidence.py').read_bytes()).hexdigest()}

if __name__=='__main__':
    path=Path('research/v8-evidence/envelope-audit.json')
    if path.exists():raise ValueError('Refusing overwrite')
    result=run();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(bool(result['failures']))
