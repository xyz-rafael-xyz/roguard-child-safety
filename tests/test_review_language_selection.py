import unittest
from training.select_review_languages_v8 import select_epochs


def entry(epoch,ro_false,uk_false):
    return {'epoch':epoch,'development':{lang+'_'+kind:{'positive':100,'negative':100,'missed':1,'false_reviews':false} for lang,false in [('ro',ro_false),('uk',uk_false)] for kind in ('formal','conversational')}}


class ReviewLanguageSelectionTests(unittest.TestCase):
    def test_language_improvement_does_not_sacrifice_other_language(self):
        self.assertEqual(select_epochs([entry(1,20,2),entry(2,4,30)]),{'ro':2,'uk':1})

    def test_low_false_alerts_cannot_override_missed_concerns(self):
        a,b=entry(1,20,2),entry(2,0,0)
        b['development']['ro_conversational']['missed']=2
        self.assertEqual(select_epochs([a,b]),{'ro':1,'uk':2})

    def test_tie_prefers_earlier_epoch(self):
        self.assertEqual(select_epochs([entry(2,5,5),entry(1,5,5)]),{'ro':1,'uk':1})

    def test_no_eligible_candidate_fails(self):
        a=entry(1,0,0);a['development']['uk_formal']['positive']=0
        with self.assertRaises(ValueError):select_epochs([a])
