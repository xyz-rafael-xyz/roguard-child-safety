"""Independently recount corpus structure, policy targets, tokens and leakage."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import unicodedata

from roguard.assurance import digest
from roguard.trajectory_data import file_sha
from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens
from training.semantic_diversity_v10 import CONTEXTS, OPERATORS


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def verify_population(rows, manifest, protected, tokenize):
    assertions = ('AFFIRMED', 'NEGATED', 'UNCERTAIN', 'RETRACTED')
    pairs = {p['family']:p for p in manifest['pairs']}
    if len(pairs) != len(manifest['pairs']) or not pairs:
        raise ValueError('Duplicate or empty family inventory')
    if (manifest['partition'] != 'train' or manifest['synthetic_only'] is not True
            or manifest['independently_annotated'] is not False):
        raise ValueError('Manifest provenance differs')
    expected = set()
    for lang, family, concern in itertools.product(('ro','uk'),pairs,(0,1)):
        expected.add((lang,family,concern,'direct','AFFIRMED',0,'bare',True))
        for scope, assertion, op, place in itertools.product(('direct','fiction'),assertions,(0,1),('before','after')):
            expected.add((lang,family,concern,scope,assertion,op,place,False))
    old_tokens = set(); old_payloads = set(); old_focus = set()
    for r in protected:
        if r['partition'] not in ('train','dev'):
            raise ValueError('Reserved partition encountered')
        old_tokens.add(r['token_sha256']); old_payloads.add(digest(r['payload']))
        old_focus.add((r['language'],normalize(r['payload']['turns'][-1]['text'])))
    cores = {(lang,normalize(core)) for p in pairs.values() for lang in ('ro','uk') for core in p[lang]}
    if len(cores) != len(pairs)*4:
        raise ValueError('Core inventory is not distinct')
    for lang, text in old_focus:
        if any(language==lang and core in text for language,core in cores):
            raise ValueError('Existing input contains an authored core')
    seen = set(); inputs = set(); tokens_seen = set(); counts = Counter()
    for row in rows:
        f=row['factors']
        key=tuple(f[k] for k in ('language','family','concern','scope','assertion','operator','placement','bare'))
        if key not in expected or key in seen:
            raise ValueError('Unexpected or repeated factor combination')
        seen.add(key)
        lang,family,concern,scope,assertion,op,place,bare=key
        core=pairs[family][lang][concern]
        attrs=dict(subject='SELF' if scope=='direct' else 'FICTIONAL',time='CURRENT',
                   assertion=assertion,source='DIRECT' if scope=='direct' else 'FICTION',
                   relation='RETRACTS' if assertion=='RETRACTED' else 'NEW',concern=str(concern))
        fields=[0 if scope=='direct' else 2,0,assertions.index(assertion),
                0 if scope=='direct' else 2,int(assertion=='RETRACTED'),concern]
        gold=int(concern==1 and (assertion=='RETRACTED' or (scope=='direct' and assertion!='NEGATED')))
        if bare:
            turns=[dict(role='minor',text=core)]
        else:
            quote='«'+core+'»'; operator=OPERATORS[lang][assertion][op]
            final=(operator+' '+quote) if place=='before' else (quote+' '+operator)
            turns=[dict(role='minor',text=core if assertion=='RETRACTED' else CONTEXTS[lang]['start']),
                   dict(role='assistant',text=CONTEXTS[lang][scope]),dict(role='minor',text=final)]
        payload=dict(language=lang,turns=turns)
        checks={'case_id':'semantic99:'+digest(f),'language':lang,'partition':'train',
                'group':'authored-semantic-diversity','family':family,'attributes':attrs,
                'fields':fields,'gold':gold,'payload':payload,'source_focus':len(turns)-1,
                'core':core,'synthetic_only':True,'independently_annotated':False}
        if any(row.get(k)!=v for k,v in checks.items()):
            raise ValueError('Generated content, provenance or label differs')
        token_values=tokenize(payload)
        token_sha=digest(token_values)
        if (row['tokens']!=token_values or row['token_count']!=len(token_values)
                or row['token_sha256']!=token_sha
                or token_values[:len(PREFIX_TOKENS)]!=PREFIX_TOKENS
                or not len(PREFIX_TOKENS)<len(token_values)<=600):
            raise ValueError('Runtime token replay differs')
        payload_sha=digest(payload)
        if (payload_sha in inputs or payload_sha in old_payloads or token_sha in tokens_seen
                or token_sha in old_tokens or (lang,normalize(turns[-1]['text'])) in old_focus):
            raise ValueError('Duplicate or protected input')
        inputs.add(payload_sha);tokens_seen.add(token_sha)
        counts[(lang,scope,assertion,str(concern),gold,bare)]+=1
    if seen!=expected:
        raise ValueError('Missing registered factor combinations')
    return dict(rows=len(rows),families=len(pairs),
                distinct_cores={lang:sum(l==lang for l,_ in cores) for lang in ('ro','uk')},
                minimum_tokens=min(r['token_count'] for r in rows),
                maximum_tokens=max(r['token_count'] for r in rows),
                counts=[dict(language=k[0],scope=k[1],assertion=k[2],concern=k[3],gold=k[4],bare=k[5],rows=n)
                        for k,n in sorted(counts.items())])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','private_root','base','audit_plan','output'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    p.add_argument('--audit-plan-sha256',required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite audit')
    if file_sha(a.audit_plan)!=a.audit_plan_sha256:raise ValueError('Audit plan differs')
    plan=json.loads(a.audit_plan.read_text())
    if plan['source_sha256']!=file_sha(__file__):raise ValueError('Auditor differs')
    def check():
        for base,bindings in ((a.root,plan['public']),(a.private_root,plan['private'])):
            for name,sha in bindings.items():
                if file_sha(base/name)!=sha:raise ValueError('Audit binding differs: '+name)
    check()
    preparation=json.loads((a.root/plan['preparation_plan']).read_text())
    report=json.loads((a.root/plan['report']).read_text())
    for base,bindings in ((a.root,preparation['public']),(a.private_root,preparation['private'])):
        for name,sha in bindings.items():
            if file_sha(base/name)!=sha:raise ValueError('Preparation binding differs: '+name)
    if a.base.name!=preparation['tokenizer_revision']:raise ValueError('Tokenizer revision differs')
    for name,sha in preparation['tokenizer_files'].items():
        if file_sha(a.base/name)!=sha:raise ValueError('Tokenizer differs')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    manifest=json.loads((a.private_root/preparation['manifest']).read_text())
    rows=[json.loads(line) for line in (a.private_root/plan['rows']).open()]
    protected=[json.loads(line) for line in (a.private_root/preparation['protected_inputs']).open()]
    result=verify_population(rows,manifest,protected,lambda payload:classifier_tokens(tokenizer,payload))
    if (not report['complete'] or any(report[k]!=v for k,v in result.items())
            or report['rows_sha256']!=file_sha(a.private_root/plan['rows'])
            or report['plan_sha256']!=file_sha(a.root/plan['preparation_plan'])
            or report['source_sha256']!=preparation['public']['training/prepare_semantic_diversity_v10.py']
            or len(rows)!=preparation['rows'] or result['families']!=preparation['families']):
        raise ValueError('Preparation report differs from independent recount')
    for key in ('training_admitted','training_performed','independently_annotated','held_out_test_opened','qualified'):
        if report[key] is not False:raise ValueError('Unsupported preparation claim')
    check()
    result.update(verification='PASS',source_sha256=file_sha(__file__),
                  audit_plan_sha256=a.audit_plan_sha256,rows_sha256=file_sha(a.private_root/plan['rows']),
                  actual_runtime_token_replays=len(rows),encoder_loaded=False,training_admitted=False,
                  independent_language_or_label_validation=False,held_out_test_opened=False,qualified=False,
                  scope='Independent structure/limited-world policy recount and full tokenizer replay. Shared registered operator/context strings are checked as data. This does not independently validate natural-language meaning or concern labels.')
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('verification','rows','families','actual_runtime_token_replays','minimum_tokens','maximum_tokens','training_admitted')}))


if __name__=='__main__':main()
