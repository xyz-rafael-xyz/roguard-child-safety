"""Blinded annotation packets and measured reviewer-study summaries.

Pseudonymous IDs are declarations, not authenticated proof of independence.
No network, invitations, or contact with participants is performed here.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random

from .assurance import digest
from .trajectory_data import validate_case, prefix, input_identity
from .trajectory_evaluation import checkpoint_id


def annotation_packet(cases, *, annotator_id, seed=1010):
    if not isinstance(annotator_id,str) or not annotator_id.strip():
        raise ValueError('A pseudonymous annotator ID is required')
    rows=[];key={};seen=set()
    for case in cases:
        validate_case(case)
        if case['case_id'] in seen:raise ValueError('Duplicate annotation case')
        seen.add(case['case_id'])
        for point in case['checkpoints']:
            payload=prefix(case,point['focus'])
            token=digest([seed,annotator_id,checkpoint_id(case,point)])[:24]
            rows.append({'item_id':token,'conversation':payload,'input_digest':input_identity(payload),
                         'review':None,'world':None,'outside_ontology':None,'evidence':[],
                         'uncertain':None,'notes':''})
            key[token]={'checkpoint_id':checkpoint_id(case,point),'input_digest':input_identity(payload)}
    random.Random(seed).shuffle(rows)
    packet={'format':'roguard-v10-blind-annotation-1','annotator_id':annotator_id,
            'instructions':'Label only the displayed prefix. Null means unresolved, never negative. Evidence offsets use Unicode code points. Do not infer event truth.',
            'items':rows}
    return packet, {'packet_digest':digest(packet),'annotator_id':annotator_id,'key':key}


def validate_annotations(packet, key, completed):
    if key['packet_digest']!=digest(packet) or completed.get('annotator_id')!=packet['annotator_id']:
        raise ValueError('Annotation packet identity mismatch')
    expected={r['item_id']:r for r in packet['items']}
    seen=set();labels={}
    for row in completed['items']:
        token=row['item_id']
        if token not in expected or token in seen:raise ValueError('Unknown or duplicate annotation')
        seen.add(token);source=expected[token]
        if row['conversation']!=source['conversation'] or row['input_digest']!=source['input_digest']:
            raise ValueError('Annotator changed the source input')
        if type(row['uncertain']) is not bool or type(row['outside_ontology']) is not bool:
            raise ValueError('Complete uncertainty and ontology declarations')
        review=row['review']
        if review is not None and (type(review) is not int or review not in (0,1)):
            raise ValueError('Invalid human review label')
        if review is None and not row['uncertain']:
            raise ValueError('Missing label must remain uncertain')
        probe={'format':'roguard-trajectory-v10-research1','case_id':token,**source['conversation'],
               'split':'test','family':token,'author':'pending','generator':'human','synthetic':True,
               'checkpoints':[{'focus':len(source['conversation']['turns'])-1,'review':review if review is not None else 0,
                               'world':row['world'],'evidence':row['evidence']}]}
        validate_case(probe)
        if row['outside_ontology'] and row['world'] is not None:
            raise ValueError('Outside-ontology annotation cannot supply a known world')
        if not row['uncertain'] and not row['outside_ontology'] and row['world'] is None:
            raise ValueError('A resolved in-ontology annotation requires a complete world')
        if not isinstance(row['notes'],str) or len(row['notes'])>10000:
            raise ValueError('Annotation notes must be bounded text')
        labels[key['key'][token]['checkpoint_id']]={k:deepcopy(row[k]) for k in ('review','world','outside_ontology','evidence','uncertain','notes')}
    if seen!=set(expected):raise ValueError('Incomplete annotation return')
    return labels


def annotation_agreement(left, right):
    if not left or set(left)!=set(right):raise ValueError('Annotation identities differ')
    conflicts=[];resolved=0;agree=0
    for key in left:
        a,b=left[key],right[key]
        if not a['uncertain'] and not b['uncertain'] and a['review'] is not None and b['review'] is not None:
            resolved+=1;agree+=a['review']==b['review']
        if any(a[k]!=b[k] for k in ('review','world','outside_ontology','evidence','uncertain')):
            conflicts.append(key)
    return {'items':len(left),'jointly_resolved':resolved,'binary_agreement':agree/resolved if resolved else None,
            'requires_adjudication':conflicts,'independence_verified':False}


def reviewer_packet(cases, predictions_a, predictions_b, *, reviewer_id, arm=0, seed=1010):
    """Each reviewer sees each conversation once; the paired reviewer swaps arms."""
    from .trajectory_evaluation import evaluate
    evaluate(cases,predictions_a,bootstrap=0);evaluate(cases,predictions_b,bootstrap=0)
    if arm not in (0,1) or type(arm) is not int:raise ValueError('arm must be 0 or 1')
    models=[{r['checkpoint_id']:r for r in predictions_a},{r['checkpoint_id']:r for r in predictions_b}]
    shuffled=list(cases);random.Random(seed).shuffle(shuffled)
    rows=[];mapping={}
    for index,case in enumerate(shuffled):
        condition=(index+arm)%2
        point=max(case['checkpoints'],key=lambda p:p['focus'])
        key=checkpoint_id(case,point);result=models[condition][key]
        token=digest([reviewer_id,key,seed])[:24]
        proposal={k:deepcopy(result[k]) for k in ('predicted','evidence') if k in result}
        # Avoid model names, digests and descriptive error strings that unblind.
        rows.append({'item_id':token,'conversation':prefix(case,point['focus']),
                     'proposal':proposal,'review':None,'elapsed_seconds':None,'task_complete':None})
        mapping[token]={'checkpoint_id':key,'condition':'A' if condition==0 else 'B','gold':point['review'],
                        'family':case['family'],'language':case['language']}
    packet={'format':'roguard-v10-reviewer-study-1','reviewer_id':reviewer_id,'items':rows,
            'instructions':'Review the proposed evidence and decide whether review is warranted. Record active elapsed time; leave interrupted tasks incomplete.'}
    return packet,{'packet_digest':digest(packet),'key':mapping,'reviewer_id':reviewer_id}


def score_reviewer(packet,key,completed):
    if digest(packet)!=key['packet_digest'] or completed.get('reviewer_id')!=packet['reviewer_id']:
        raise ValueError('Reviewer packet mismatch')
    original={r['item_id']:r for r in packet['items']};seen=set();groups={'A':[],'B':[]}
    for row in completed['items']:
        token=row['item_id']
        if token not in original or token in seen:raise ValueError('Unknown or duplicate reviewer item')
        seen.add(token)
        if any(row[k]!=original[token][k] for k in ('conversation','proposal')):
            raise ValueError('Reviewer input changed')
        if type(row['task_complete']) is not bool:raise ValueError('Missing completion declaration')
        if row['task_complete']:
            if type(row['review']) is not int or row['review'] not in (0,1):raise ValueError('Missing review decision')
            t=row['elapsed_seconds']
            if type(t) not in (int,float) or not math.isfinite(t) or not 0<t<=86400:raise ValueError('Invalid elapsed review time')
        groups[key['key'][token]['condition']].append((row,key['key'][token]))
    if seen!=set(original):raise ValueError('Missing reviewer items')
    summary={}
    for condition,rows in groups.items():
        done=[(r,k) for r,k in rows if r['task_complete']]
        summary[condition]={'assigned':len(rows),'completed':len(done),
            'accuracy_including_incomplete':sum(r['review']==k['gold'] for r,k in done)/len(rows) if rows else None,
            'mean_seconds_completed':sum(r['elapsed_seconds'] for r,k in done)/len(done) if done else None}
    return {'reviewer_id':packet['reviewer_id'],'conditions':summary,'independence_verified':False,
            'causal_benefit_established':False,'note':'Descriptive single-reviewer result; aggregate the preregistered counterbalanced study.'}


def write_kit(output, cases):
    output=Path(output)
    if output.exists():raise ValueError('Refusing to overwrite a reviewer kit')
    output.mkdir(parents=True)
    for annotator in ('annotator-a','annotator-b'):
        packet,key=annotation_packet(cases,annotator_id=annotator)
        (output/(annotator+'.json')).write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
        (output/(annotator+'-PRIVATE-key.json')).write_text(json.dumps(key,indent=2)+'\n')
    (output/'README.txt').write_text('Distribute only the annotator JSON to its assigned annotator. Keep PRIVATE-key files with the study coordinator. These practice cases are developer-authored, not the independent sealed test. The source code cannot authenticate authors or reviewers.\n')
