import gzip
import json
from pathlib import Path
import tempfile
import unittest
from training.compare_review_candidates_v9 import compare


class ReviewComparisonTests(unittest.TestCase):
    def test_rejects_changed_populations_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            a.mkdir(); b.mkdir()
            row = {'case_id':'one','language':'ro','turns':[{'role':'minor','text':'Salut.'}], 'focus':0,
                   'attributes':{'subject':'SELF','source':'DIRECT','time':'CURRENT','assertion':'AFFIRMED','relation':'NEW','concern':'0'}}
            pred = {'case_id':'one','language':'ro','gold':0,'predicted':0}
            for d in (a,b):
                for split in ('dev','cal'):
                    (d/(split+'.jsonl.gz')).write_bytes(gzip.compress((json.dumps(row)+'\n').encode()))
                    (d/(split+'-predictions.jsonl')).write_text(json.dumps(pred)+'\n')
            self.assertEqual(compare(a,b,'ro')['splits']['dev']['groups']['ro_formal']['regressed'],0)
            changed = dict(row, turns=[{'role':'minor','text':'Bună.'}])
            (b/'dev.jsonl.gz').write_bytes(gzip.compress((json.dumps(changed)+'\n').encode()))
            with self.assertRaisesRegex(ValueError,'populations'):
                compare(a,b,'ro')
            (b/'dev.jsonl.gz').write_bytes(gzip.compress((json.dumps(row)+'\n').encode()))
            (b/'dev-predictions.jsonl').write_text(json.dumps(dict(pred,gold=1))+'\n')
            with self.assertRaisesRegex(ValueError,'gold'):
                compare(a,b,'ro')
            (b/'dev-predictions.jsonl').write_text((json.dumps(pred)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'coverage'):
                compare(a,b,'ro')

    def test_mixture_selection_does_not_use_calibration_scores(self):
        from training.mix_review_v9 import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            a.mkdir(); b.mkdir()
            for split in ('dev','cal'):
                rows=[]
                for native in (False,True):
                    for gold in (0,1):
                        key=f'{split}-{native}-{gold}'
                        rows.append({'case_id':key,'language':'ro','turns':[{'role':'minor','text':key}], 'focus':0,
                                     'source':'v7_training_frame_projection' if native else 'synthetic',
                                     'attributes':{'subject':'SELF','source':'DIRECT','time':'CURRENT','assertion':'AFFIRMED','relation':'NEW','concern':str(gold)}})
                for folder in (a,b):
                    predictions=[]
                    for row in rows:
                        gold=int(row['attributes']['concern'])
                        score=(.8 if gold else .95) if folder==a else (.9 if gold else .1)
                        predictions.append({'case_id':row['case_id'],'language':'ro','gold':gold,'score':score,'predicted':int(score>=.5)})
                    (folder/(split+'.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(r)+'\n' for r in rows).encode()))
                    (folder/(split+'-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
            first=evaluate(a,b,'ro')
            path=b/'cal-predictions.jsonl'
            changed=[dict(json.loads(line),score=0.99) for line in path.read_text().splitlines()]
            path.write_text(''.join(json.dumps(r)+'\n' for r in changed))
            second=evaluate(a,b,'ro')
            self.assertEqual(first['selected'],second['selected'])
            self.assertNotEqual(first['calibration'],second['calibration'])
            self.assertGreater(first['selected']['candidate_weight'],0)
