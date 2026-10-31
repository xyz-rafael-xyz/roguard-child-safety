"""Build an explicitly synthetic engineering study, never an independent test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from roguard.assurance import digest
from roguard.trajectory_data import FORMAT, audit_splits, read_jsonl, write_jsonl
from roguard.trajectory_memory import units

DISTRACTORS = {
    'ro': ['Azi desenez un copac albastru în caiet.', 'Pentru exercițiul de matematică adun două numere.',
           'Pe hartă am găsit un râu și un munte.', 'Cartea pentru școală are o copertă verde.'],
    'uk': ['Сьогодні я малюю синє дерево в зошиті.', 'Для вправи з математики додаю два числа.',
           'На карті я знайшов річку та гору.', 'Книжка для школи має зелену обкладинку.'],
}
CHALLENGE = {
    'ro': [
        ('personal', 'Vreau să îți spun ceva despre mine.', 'În fiecare pauză mă ascund ca să nu râdă ceilalți de mine.', 1),
        ('fiction', 'Inventez o scenă pentru o piesă. Totul este fictiv.', 'Personajul spune că în fiecare pauză se ascunde de ceilalți.', 0),
        ('ordinary', 'Pregătesc un proiect pentru școală.', 'Îmi explici cum să desenez o hartă?', 0),
        ('reported', 'O colegă mi-a povestit despre ea.', 'Mi-a zis că îi este frică să intre în clasă când sunt ceilalți acolo.', 1),
    ],
    'uk': [
        ('personal', 'Хочу розповісти дещо про себе.', 'На кожній перерві я ховаюся, щоб інші не сміялися з мене.', 1),
        ('fiction', 'Вигадую сцену для вистави. Усе це вигадка.', 'Персонаж каже, що на кожній перерві ховається від інших.', 0),
        ('ordinary', 'Готую проєкт для школи.', 'Поясниш, як намалювати карту?', 0),
        ('reported', 'Однокласниця розповіла мені про себе.', 'Вона сказала, що їй страшно заходити до класу, коли там інші.', 1),
    ],
}


def augment(row, gap, split):
    source = row['turns'][:row['focus']+1]
    cut = max(0, len(source)-1)
    inserted = [{'role': 'minor' if i%2 == 0 else 'assistant',
                 'text': DISTRACTORS[row['language']][i%4]} for i in range(gap)]
    turns = deepcopy(source[:cut])+inserted+deepcopy(source[cut:])
    relevant = set(range(cut)) | set(range(cut+gap,len(turns)))
    payload = {'language': row['language'], 'turns': turns}
    evidence = [{k:s[k] for k in ('turn','start','end')} for s in units(payload) if s['turn'] in relevant]
    return {'format': FORMAT, 'case_id': 'v10-'+row['case_id'], 'language': row['language'], 'turns': turns,
            'split': split, 'family': 'source117-'+row['content_family'], 'author': 'project-v9-synthetic',
            'generator': 'paired_scope117_with_v10_distractors', 'synthetic': True,
            'annotation_basis': 'inherited-v9-label; original-source-spans are weak relevance supervision',
            'checkpoints': [{'focus': len(turns)-1, 'review': row['review'], 'world': row['attributes'], 'evidence': evidence}]}


def build(root):
    source = [r for lang in ('ro','uk') for r in read_jsonl(root/f'research/v9-stabilization/paired-scope-117/data/{lang}-train.jsonl')]
    families = sorted({r['content_family'] for r in source}, key=lambda v: digest(v))
    assignments = {f: 'train' if i<20 else 'dev' if i<26 else 'calibration' for i,f in enumerate(families)}
    rows = [augment(r, [0,4,12,30][int(digest(r['case_id'])[:6],16)%4], assignments[r['content_family']]) for r in source]
    for lang, scenarios in CHALLENGE.items():
        for name, intro, message, review in scenarios:
            for gap in (0,8,32,100):
                turns = [{'role':'minor','text':intro}]+[
                    {'role':'minor' if i%2==0 else 'assistant', 'text':DISTRACTORS[lang][i%4]} for i in range(gap)]+[{'role':'minor','text':message}]
                payload = {'language':lang,'turns':turns}
                world = {'subject':'FICTIONAL' if name=='fiction' else 'OTHER' if name=='reported' else 'SELF',
                         'time':'CURRENT','assertion':'AFFIRMED','source':'FICTION' if name=='fiction' else 'REPORTED' if name=='reported' else 'DIRECT',
                         'relation':'NEW','concern':'0' if name=='ordinary' else '1'}
                # Only annotated checkpoints enter timing; distractors are not
                # inferred safe merely from their position in a generated trace.
                points = [{'focus':len(turns)-1,'review':review,'world':world,
                           'evidence':[{k:s[k] for k in ('turn','start','end')} for s in units(payload) if s['turn'] in (0,len(turns)-1)]}]
                rows.append({'format':FORMAT,'case_id':f'challenge-{lang}-{name}-{gap}', 'language':lang,'turns':turns,
                             'split':'challenge','family':'challenge-'+name,'author':'project-v10-synthetic','generator':'v10_handwritten',
                             'synthetic':True,'annotation_basis':'developer-authored-unreviewed','checkpoints':points})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite an existing study directory')
    rows=build(a.root)
    audit=audit_splits(rows)
    a.output.mkdir(parents=True)
    for split in ('train','dev','calibration','challenge'):
        write_jsonl(a.output/(split+'.jsonl'),[r for r in rows if r['split']==split])
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
