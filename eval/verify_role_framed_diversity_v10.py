"""Verify allowed parent-to-continuation changes without calling the transform."""
from collections import Counter

from roguard.assurance import digest
from roguard.gemma4_contract import PREFIX_TOKENS
from training.role_framed_diversity_v10 import FRAMES


def verify_rows(parents, rows, tokenize, protected):
    by_id={r['case_id']:r for r in parents}
    if len(by_id)!=len(parents) or not parents or len(rows)!=len(parents):
        raise ValueError('Parent population differs')
    old_tokens={r['token_sha256'] for r in protected}
    old_payloads={digest(r['payload']) for r in protected}
    if any(r['partition'] not in ('train','dev') for r in protected):
        raise ValueError('Reserved partition encountered')
    seen=set(); token_seen=set(); payload_seen=set(); counts=Counter()
    allowed={'case_id','group','parent_case_id','parent_row_sha256','framing_changed',
             'payload','tokens','token_sha256','token_count'}
    for row in rows:
        parent_id=row['parent_case_id']
        if parent_id not in by_id or parent_id in seen:raise ValueError('Missing or repeated parent')
        seen.add(parent_id);parent=by_id[parent_id]
        if (set(row)!=set(parent)|{'parent_case_id','parent_row_sha256','framing_changed'}
                or any(row[k]!=v for k,v in parent.items() if k not in allowed)):
            raise ValueError('Protected label, core, factor or metadata changed')
        fiction=parent['factors']['scope']=='fiction'
        if (row['case_id']!='semantic100:'+parent_id.removeprefix('semantic99:')
                or row['parent_row_sha256']!=digest(parent)
                or row['group']!='role-framed-authored-semantic-diversity'
                or row['framing_changed'] is not fiction):
            raise ValueError('Continuation provenance differs')
        before=parent['payload'];after=row['payload']
        if fiction:
            if set(before)!=set(after) or after['language']!=before['language']:
                raise ValueError('Payload envelope changed')
            a=before['turns'];b=after['turns'];frame=FRAMES[parent['language']]
            if (len(a)!=3 or len(b)!=3 or a[1]!=b[1]
                    or any(set(t)!={'role','text'} for t in b)
                    or [t['role'] for t in b]!=['minor','assistant','minor']):
                raise ValueError('Conversation roles/context changed')
            first=frame['prior_quote'].format(core=parent['core']) if parent['attributes']['assertion']=='RETRACTED' else frame['prior']
            if b[0]['text']!=first or b[2]['text']!=frame['focus']+a[2]['text']:
                raise ValueError('Explicit minor framing differs')
            if after==before:raise ValueError('Fictional input was not revised')
        elif after!=before:
            raise ValueError('Direct payload changed')
        values=tokenize(after);checksum=digest(values);key=digest(after)
        if (row['tokens']!=values or row['token_sha256']!=checksum or row['token_count']!=len(values)
                or values[:len(PREFIX_TOKENS)]!=PREFIX_TOKENS
                or not len(PREFIX_TOKENS)<len(values)<=600):
            raise ValueError('Runtime tokens differ')
        if not fiction and any(row[k]!=parent[k] for k in ('tokens','token_sha256','token_count')):
            raise ValueError('Direct token sequence changed')
        if key in payload_seen or key in old_payloads or checksum in token_seen or checksum in old_tokens:
            raise ValueError('Duplicate or protected model input')
        token_seen.add(checksum);payload_seen.add(key)
        counts[(row['language'],fiction)]+=1
    if seen!=set(by_id):raise ValueError('Incomplete parent coverage')
    return dict(rows=len(rows),fiction_rows_changed=sum(n for (_,f),n in counts.items() if f),
                direct_rows_unchanged=sum(n for (_,f),n in counts.items() if not f),
                labels_cores_factors_preserved=True,actual_token_replays=len(rows),
                minimum_tokens=min(r['token_count'] for r in rows),maximum_tokens=max(r['token_count'] for r in rows),
                counts=[dict(language=k[0],fiction=k[1],rows=n) for k,n in sorted(counts.items())])
