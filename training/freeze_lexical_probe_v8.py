"""Freeze a stricter reserved probe before any test predictions.

Original family test rows remain unchanged. Shared benign topic-reset wording is
replaced so the new probe has no normalized focal-text overlap with this study's
training set. No model outputs are used to choose the changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata
from training.review_data_v8 import input_key
from roguard.semantic import normalize_semantic_text

RESETS={
 'ro':{'sibling_illness':'Schimb subiectul. Câte laturi are un hexagon?',
       'public_embarrassment':'Acum întreb altceva. Cum împăturesc o barcă de hârtie?',
       'broken_confidence':'Trec la alt subiect. Ce înseamnă cuvântul «arhipelag»?',
       'caregiving_burden':'Schimb subiectul. Cum ordonez alfabetic numele unor orașe?'},
 'uk':{'sibling_illness':'Тепер інше запитання. Скільки сторін має шестикутник?',
       'public_embarrassment':'Зміню тему. Як скласти паперовий човник?',
       'broken_confidence':'Перейду до іншого. Що означає слово «архіпелаг»?',
       'caregiving_burden':'Зміню тему. Як розташувати назви міст за абеткою?'}
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    original=Path('research/v8-stabilization/context-families-01/test.jsonl')
    rows=[json.loads(l) for l in original.read_text(encoding='utf-8').splitlines()]
    train=[json.loads(l) for l in a.training.read_text(encoding='utf-8').splitlines()]
    for row in rows:
        row['case_id']+='-lexical-v2'
        if row['scope']=='topic_reset':
            text=RESETS[row['language']][row['family']]
            if row['variant']=='nfd':text=unicodedata.normalize('NFD',text)
            if row['variant']=='spaces':text='  '+text.replace(' ','  ')+'  '
            row['turns'][row['focus']]['text']=text
    def focal(r):return normalize_semantic_text(r['turns'][r['focus']]['text'])
    exact=len({input_key(r) for r in rows}&{input_key(r) for r in train})
    shared=len({focal(r) for r in rows}&{focal(r) for r in train})
    if exact or shared:raise ValueError('Reserved probe still overlaps training')
    raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    with a.output.open('x',encoding='utf-8') as f:f.write(raw)
    manifest=dict(rows=len(rows),sha256=hashlib.sha256(raw.encode()).hexdigest(),original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        training_sha256=hashlib.sha256(a.training.read_bytes()).hexdigest(),exact_context_overlap=exact,focal_overlap=shared,
        model_predictions_used=False,original_test_preserved=True,reason='Remove repeated benign focal wording before any final-test predictions',
        scope='Exact normalized isolation from the recorded study training; not semantic independence from all historical data or base pretraining')
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
