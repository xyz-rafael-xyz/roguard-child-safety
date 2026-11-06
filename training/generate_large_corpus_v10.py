"""Resumable, bound bilingual scenario generation; no training-ready label claim."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import unicodedata

from roguard.assurance import digest
from roguard.trajectory_data import file_sha

THEMES = ('dependence on a chatbot instead of human support', 'isolation from friends',
          'bullying in a school club', 'pressure from someone at home', 'school performance stress',
          'pressure in an online game', 'intimidation by a peer', 'worries about eating or appearance',
          'coping with a friendship ending', 'sleep disrupted by worry', 'an adult ignoring personal boundaries',
          'exclusion from a group activity', 'persistent worry affecting daily tasks', 'unwanted messages online',
          'feeling unheard at home', 'asking a trusted person for help')
SETTINGS = ('a homework project', 'a group chat', 'a sports activity', 'an art activity',
            'a school trip', 'a weekend plan', 'a hobby club', 'an online game',
            'a shared tablet', 'a family meal', 'a library visit', 'a music rehearsal',
            'a birthday plan', 'a video call', 'a team task', 'an ordinary afternoon',
            'a school play', 'a language lesson', 'a bus journey', 'a craft fair',
            'a swimming lesson', 'a family board game', 'a school lunch break', 'a neighborhood walk',
            'a photography hobby', 'a class presentation', 'a dance lesson', 'a pet-care routine',
            'a school fundraiser', 'a holiday visit', 'a stationery shop', 'a community cleanup',
            'a bicycle inspection', 'a science exhibition', 'a garden club', 'a local museum',
            'a beach trip', 'a school newspaper', 'a chess club', 'a puppet show',
            'a repair workshop', 'a baking club', 'an astronomy outing', 'a book exchange',
            'a geography quiz', 'a nature trail', 'a ceramics class', 'a train station')
STYLES = ('plain conversational wording', 'polite but indirect wording', 'two short connected sentences',
          'a first-person statement of intention', 'a short conditional sentence',
          'a correction contrasting two possibilities', 'a leading question', 'a concrete future proposal')
BINARY_STYLES = ('plain first-person present tense', 'understated but explicit first-person wording',
                 'two short connected present-tense statements', 'a concrete current daily routine',
                 'a current action and its effect on daily functioning', 'informal but clear first-person wording',
                 'a current feeling supported by a concrete event', 'a request for help explaining a current experience')
CATEGORIES = ('boundary', 'privacy', 'pressure', 'unsupported')
DEFINITIONS = {
    'boundary': 'asks for secrecy about the chatbot relationship, exclusivity, or replacement of human relationships',
    'privacy': 'asks the user to reveal private identifying information or credentials to the assistant; do not include actual identifiers',
    'pressure': 'explicitly disregards the user refusing to disclose or wanting to stop; do not introduce private-data or relationship-exclusivity demands',
    'unsupported': 'claims certainty or guarantees an outcome despite explicitly lacking enough evidence; no medical, legal or financial advice',
}
KEYS = {'binary_review': ('ro_concern', 'ro_benign', 'uk_concern', 'uk_benign'),
        'response_behavior': ('ro_problematic', 'ro_appropriate', 'uk_problematic', 'uk_appropriate')}
SYSTEM = (
    'Create fictional, non-graphic examples for defensive conversation-classifier research. '
    'Return only the requested JSON object. Use natural Romanian with diacritics and natural Ukrainian, '
    'not Russian. The language pairs must express the same intended meaning. Each value must be one '
    'or two concise sentences, between 20 and 240 characters. No names, actual contact details, '
    'instructions for wrongdoing or harm, explicit sexual content, markdown, explanations or labels inside the sentences. '
    'Write a fresh concrete scenario, avoiding repetitive boilerplate. The examples are training candidates, not advice.'
)


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def manifest(plan):
    result = []; quotas = plan['quotas_per_task']; total = sum(quotas.values())
    partitions = [name for name, count in quotas.items() for _ in range(count)]
    for i in range(total):
        for task in plan['tasks']:
            result.append({'family': f'{plan.get("family_prefix", "bulk50")}:{task}:{i:04d}', 'partition': partitions[i], 'task': task,
                           'category': CATEGORIES[i % 4] if task == 'response_behavior' else None,
                           'theme': THEMES[i % len(THEMES)], 'setting': SETTINGS[(i//len(THEMES)) % len(SETTINGS)],
                           'style': (BINARY_STYLES if task=='binary_review' else STYLES)[(i//4) % len(STYLES)],
                           'variation': i//(len(THEMES)*len(SETTINGS))})
    return result


def prompt(request):
    if request['task'] == 'binary_review':
        task = ('Write a teenager\'s direct first-person description of a CURRENT concern that warrants a human review, '
                'and a matched ordinary BENIGN experience in the same setting. Neither is fiction-within-fiction, '
                'a quotation, a hypothetical concern, a denial, or a report about somebody else. '
                'The benign sentence must describe ordinary functioning without distress, coercion or dependence. '
                'The concern must be concrete, not just the word worried. Theme: '+request['theme']+'. ')
    else:
        task = ('Write a proposed assistant reply that '+DEFINITIONS[request['category']]+'. '
                'Pair it with a respectful alternative about the same situation which does not have any of these '
                'four problems: relationship exclusivity/secrecy, private-data requests, pressure after refusal, '
                'or unsupported certainty. Both sentences must be the assistant speaking directly to the user, '
                'not a user quoting bad advice or an analysis of someone else\'s behavior. ')
    return task+'Setting: '+request['setting']+'. Style: '+request['style']+'. Variation number: '+str(request['variation'])+'. Exactly these four keys: '+', '.join(KEYS[request['task']])+'.'


def json_object(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict): raise ValueError('not_json_object')
    return value


def combine_native(parts, request):
    if set(parts) != {'ro', 'uk'}: raise ValueError('missing_native_language')
    values = {}
    for language, part in parts.items():
        value = json_object(part)
        expected = {key[3:] for key in KEYS[request['task']] if key.startswith(language+'_')}
        if set(value) != expected: raise ValueError('wrong_native_schema')
        values.update({language+'_'+key: text for key, text in value.items()})
    return json.dumps(values, ensure_ascii=False)


def parse(raw, request):
    value = json_object(raw)
    if not isinstance(value, dict) or set(value) != set(KEYS[request['task']]): raise ValueError('wrong_schema')
    cleaned = {}
    for key, text in value.items():
        if not isinstance(text, str): raise ValueError('non_string')
        text = unicodedata.normalize('NFC', text).strip()
        if not 20 <= len(text) <= 240 or any(ord(c) < 32 for c in text): raise ValueError('text_bounds')
        letters = [c for c in text if c.isalpha()]
        script = 'LATIN' if key.startswith('ro_') else 'CYRILLIC'
        if len(letters) < 12 or sum(script in unicodedata.name(c, '') for c in letters)/len(letters) < .9:
            raise ValueError('wrong_writing_system')
        cleaned[key] = text
    if len({normalized(t) for t in cleaned.values()}) != 4: raise ValueError('duplicate_pair_text')
    return cleaned


def protected_texts():
    root = Path('research/v10-evidence')
    paths = [root/'response-curriculum-25'/(name+'.jsonl') for name in ('train','dev','old-diagnostic')]
    paths += [root/'contrast-response-32/additional-train.jsonl', root/'response-context-34/inputs.jsonl']
    paths += list(sorted((root/'data').glob('*.jsonl')))
    paths += [root/'broad-v9-regression/cases.jsonl', root/'committee-challenge-data-44/cases.jsonl', root/'robust-training-47/train.jsonl']
    paths += [Path('research/v9-stabilization')/name/'inputs.jsonl.gz' for name in
              ('shared-tail-corpus-164','grounding-tail-corpus-174','referent-tail-corpus-176','reported-tail-corpus-195')]
    seen = set(); bindings = {}
    for path in paths:
        bindings[str(path)] = file_sha(path)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); row = row.get('row', row)
                texts = [row[name] for name in ('draft','context') if isinstance(row.get(name), str)]
                texts += [turn['text'] for turn in row.get('turns', [])]
                seen.update(normalized(text) for text in texts if text.strip())
    return seen, bindings


def append(path, record):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())


def read_records(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8') as stream: return [json.loads(line) for line in stream]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','study','private'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--limit', type=int); p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-report', type=Path)
    p.add_argument('--amendment', type=Path, default=Path('research/v10-evidence/protocol-amendment-48.json'))
    a = p.parse_args(); amendment = a.amendment
    plan = json.loads(amendment.read_text()); requests = manifest(plan)
    if a.limit is not None and not 1 <= a.limit <= len(requests): raise ValueError('Invalid family limit')
    if a.base.name != plan['generator']['revision']: raise ValueError('Unexpected generator revision')
    protected, bindings = protected_texts()
    protocol = {'format':'roguard-large-generation-1','source_sha256':file_sha(__file__),
                'amendment_sha256':file_sha(amendment),'manifest_sha256':digest(requests),
                'generator':plan['generator'],'system_prompt':SYSTEM,'historical_inputs':bindings,
                'historical_unique_texts':len(protected),'requested_families':len(requests),
                'training_ready':False,'semantic_labels_verified':False,'stable_release_ready':False}
    if plan.get('native_generation'):
        protocol['system_prompt'] = None
        protocol['language_pairing'] = 'Independent monolingual pairs share a request and partition; they are not parallel translations.'
        protocol['native_prompt_source_sha256']=file_sha('training/native_generation_prompts_v10.py')
        protocol['demonstration_source_sha256']=file_sha('training/contrast_response_data_v10.py')
    if a.preflight_report:
        if a.preflight_report.exists(): raise ValueError('Preflight exists')
        a.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        a.preflight_report.write_text(json.dumps({**protocol,'model_loaded':False},indent=2)+'\n');return
    if a.resume:
        if json.loads((a.study/'protocol.json').read_text()) != protocol: raise ValueError('Resume bindings differ')
    else:
        if a.study.exists() or a.private.exists(): raise ValueError('Refusing to overwrite generation')
        a.study.mkdir(parents=True);a.private.mkdir(parents=True)
        (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
        if plan.get('native_generation'):
            (a.study/'native-prompts.py').write_bytes(Path('training/native_generation_prompts_v10.py').read_bytes())
        (a.study/'manifest.json').write_text(json.dumps(requests,indent=2)+'\n')
    lock=(a.private/'writer.lock').open('a')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory={path.name:file_sha(path) for path in sorted(a.base.iterdir()) if path.is_file()}
    if not any(name.endswith('.safetensors') for name in inventory): raise ValueError('No generator weights')
    binding=a.study/'model-files.json'
    if binding.exists():
        if json.loads(binding.read_text()) != inventory: raise ValueError('Generator files changed')
    else: binding.write_text(json.dumps(inventory,indent=2)+'\n')
    completed=read_records(a.private/'families.jsonl'); attempts=read_records(a.private/'attempts.jsonl')
    by_id={r['request']['family']:r for r in completed}
    if len(by_id)!=len(completed): raise ValueError('Duplicate completed family')
    expected={r['family']:r for r in requests}
    by_attempt={}
    for row in attempts:
        key=(row['family'],row['attempt'])
        if key in by_attempt: raise ValueError('Duplicate attempt record')
        by_attempt[key]=row
    for row in completed:
        if row['request']!=expected.get(row['request']['family']): raise ValueError('Completed request differs')
        if row['accepted']:
            attempt=by_attempt.get((row['request']['family'],row['accepted_attempt']))
            if attempt is None or parse(attempt['raw'],row['request'])!=row['texts']:
                raise ValueError('Completed family differs from its retained generation')
            if any(normalized(text) in protected for text in row['texts'].values()):
                raise ValueError('Completed family now overlaps protected inputs')
            protected.update(normalized(text) for text in row['texts'].values())
    import mlx.core as mx
    from mlx_lm import load,generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(10*1024**3);mx.set_cache_limit(128*1024**2)
    engine,tokenizer=load(str(a.base),tokenizer_config={'trust_remote_code':False})
    if plan['generator'].get('compute_dtype','float32')=='float32':engine.set_dtype(mx.float32)
    elif plan['generator']['compute_dtype']!='native':raise ValueError('Unsupported generator dtype')
    model=getattr(engine,'language_model',engine);started=time.monotonic()
    for index,request in enumerate(requests[:a.limit]):
        if request['family'] in by_id: continue
        if shutil.disk_usage(a.private).free < 8*1024**3: raise RuntimeError('Disk reserve reached; preserve evidence and resume later')
        accepted=None;rejections=[]
        for attempt in range(plan['generator']['attempts_per_family']):
            key=(request['family'],attempt)
            if key in by_attempt: record=by_attempt[key]
            else:
                seed=plan['generator']['seed']+index*plan['generator']['attempts_per_family']+attempt;mx.random.seed(seed)
                tick=time.monotonic();parts={};query_hashes={}
                for language in (('ro','uk') if plan.get('native_generation') else ('both',)):
                    if language=='both':system,query=SYSTEM,prompt(request)
                    else:
                        from training.native_generation_prompts_v10 import native_query
                        system,query=native_query(request,language)
                    messages=[{'role':'system','content':system},{'role':'user','content':query}]
                    if plan.get('native_generation'):
                        messages.append({'role':'assistant','content':'{'})
                        tokens=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,continue_final_message=True,enable_thinking=False)
                    else:
                        tokens=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
                    if not tokens or len(tokens)>2048: raise ValueError('Generator prompt exceeds envelope')
                    parts[language]=('{' if plan.get('native_generation') else '')+generate(model,tokenizer,tokens,max_tokens=plan['generator']['maximum_output_tokens'],
                        sampler=make_sampler(temp=plan['generator']['temperature'],top_p=plan['generator']['top_p']),verbose=False)
                    query_hashes[language]=digest(tokens);mx.clear_cache()
                raw=parts.get('both')
                if raw is None:
                    try:
                        raw=combine_native(parts,request)
                    except (ValueError,TypeError,AttributeError):raw=''
                record={'family':request['family'],'attempt':attempt,'seed':seed,'query_token_sha256':query_hashes,
                        'raw':raw,'raw_parts':parts,'seconds':time.monotonic()-tick}
                append(a.private/'attempts.jsonl',record);by_attempt[key]=record;mx.clear_cache()
            try:
                values=parse(record['raw'],request)
                if any(normalized(text) in protected for text in values.values()): raise ValueError('historical_or_prior_family_exact_overlap')
                accepted=values;break
            except (ValueError,TypeError) as error:
                rejections.append({'attempt':attempt,'reason':str(error)[:120]})
                print(json.dumps({'family':request['family'],'attempt':attempt,'rejected':str(error)[:120]}),flush=True)
        result={'request':request,'accepted':accepted is not None,'accepted_attempt':attempt if accepted else None,
                'rejections':rejections,'texts':accepted,'semantic_labels_verified':False}
        append(a.private/'families.jsonl',result);by_id[request['family']]=result
        if accepted: protected.update(normalized(text) for text in accepted.values())
        progress={'completed_families':len(by_id),'accepted_families':sum(r['accepted'] for r in by_id.values()),
                  'requested_families':len(requests),'current_partition':request['partition'],
                  'seconds_this_process':time.monotonic()-started,'training_ready':False}
        temporary=a.study/'progress.tmp';temporary.write_text(json.dumps(progress,indent=2)+'\n');temporary.replace(a.study/'progress.json')
        print(json.dumps(progress),flush=True)
    records=list(by_id.values());artifacts={}
    for partition in plan['quotas_per_task']:
        path=a.private/(partition+'-families.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records if r['request']['partition']==partition),encoding='utf-8')
        artifacts[partition]=file_sha(path)
    report={'complete':len(records)==len(requests),'pilot_or_partial':len(records)!=len(requests),
            'completed_families':len(records),'accepted_families':sum(r['accepted'] for r in records),
            'counts':dict(Counter(r['request']['task']+':'+r['request']['partition']+':'+str(r['accepted']) for r in records)),
            'families_sha256':file_sha(a.private/'families.jsonl'),'attempts_sha256':file_sha(a.private/'attempts.jsonl'),
            'partition_sha256':artifacts,'training_ready':False,'semantic_labels_verified':False,'stable_release_ready':False}
    suffix='report.json' if report['complete'] else 'partial-'+str(len(records))+'.json'
    (a.study/suffix).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
