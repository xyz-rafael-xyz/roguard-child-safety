import unittest
from training.check_review_candidate_v8 import GROUPS,check_counts

class ReviewPromotionCountsTests(unittest.TestCase):
    def metrics(self):
        return {k:{'positive':100,'negative':100,'missed':1,'false_reviews':5} for k in GROUPS}

    def test_exact_boundaries_pass(self):
        self.assertEqual(check_counts(self.metrics()),[])

    def test_every_language_and_group_must_pass(self):
        for group in GROUPS:
            for field in ('missed','false_reviews'):
                with self.subTest(group=group,field=field):
                    m=self.metrics();m[group][field]+=1
                    self.assertTrue(check_counts(m))

    def test_missing_zero_and_invalid_counts_fail(self):
        for replacement in ({},{'positive':0,'negative':100,'missed':0,'false_reviews':0},
                            {'positive':100,'negative':100,'missed':-1,'false_reviews':0},
                            {'positive':100,'negative':100,'missed':False,'false_reviews':0}):
            m=self.metrics();m['uk_conversational']=replacement
            self.assertTrue(check_counts(m))

    def test_rounded_reported_rates_cannot_override_counts(self):
        m=self.metrics();m['ro_formal'].update(positive=99,recall=.99,false_review_rate=0)
        self.assertTrue(check_counts(m))

class ReviewPromotionArtifactTests(unittest.TestCase):
    def fixture(self,root):
        import gzip,hashlib,json
        def row(split,lang,kind,positive):
            return {'case_id':f'{split}-{lang}-{kind}-{positive}','language':lang,
                    'turns':[{'role':'minor','text':f'{split} {kind} {positive}'}],'focus':0,
                    'source':'v7_training_frame_projection' if kind=='conversational' else 'formal',
                    'attributes':{'subject':'SELF','source':'DIRECT','assertion':'AFFIRMED','relation':'NEW','time':'CURRENT','concern':str(positive)}}
        for split in ('train','dev','cal'):
            rows=[row(split,lang,kind,positive) for lang in ('ro','uk') for kind in ('formal','conversational') for positive in (0,1)]
            (root/(split+'.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(r)+'\n' for r in rows).encode(),mtime=0))
        (root/'training-history.jsonl.gz').write_bytes((root/'train.jsonl.gz').read_bytes())
        (root/'review.json.gz').write_bytes(gzip.compress(b'{}',mtime=0))
        metrics={k:{'positive':1,'negative':1,'missed':0,'false_reviews':0} for k in GROUPS}
        record={'artifact_sha256':hashlib.sha256((root/'review.json.gz').read_bytes()).hexdigest(),'selected_epoch':1,'epochs':[{'epoch':1,'development':metrics}],'calibration_audit':metrics}
        (root/'development.json').write_text(json.dumps(record))

    def test_language_scope_is_not_overwritten_by_denominator_audit(self):
        import tempfile
        from pathlib import Path
        from training.check_review_candidate_v8 import inspect
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            result=inspect(root,'ro')
            self.assertEqual(result['language_scope'],'ro')
            self.assertTrue(result['study_prerequisites_pass'])
            self.assertFalse(result['stable_release_ready'])
            result=inspect(root)
            self.assertEqual(result['language_scope'],'both')
            self.assertNotIn('separately passing other-language candidate',result['additional_required_checks'])

    def test_changed_artifact_cannot_pass(self):
        import tempfile
        from pathlib import Path
        from training.check_review_candidate_v8 import inspect
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            (root/'review.json.gz').write_bytes(b'changed')
            self.assertFalse(inspect(root)['study_prerequisites_pass'])

    def test_ancestor_overlap_cannot_pass(self):
        import tempfile
        from pathlib import Path
        from training.check_review_candidate_v8 import inspect
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            (root/'training-history.jsonl.gz').write_bytes((root/'train.jsonl.gz').read_bytes()+(root/'dev.jsonl.gz').read_bytes())
            result=inspect(root)
            self.assertIn('training history/dev: overlapping inputs',result['failures'])
