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
