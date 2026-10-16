import unittest
from roguard.review_evaluation import evaluate, triage, target_failures


class ReviewEvaluationTests(unittest.TestCase):
    def test_release_targets_require_every_group(self):
        self.assertEqual(len(target_failures({'groups':{}})),4)
        groups={name:dict(positive=100,negative=100,missed_or_abstained_concerns=1,false_review_or_abstained_benign=5)
                for name in ('ro_formal','ro_conversational','uk_formal','uk_conversational')}
        self.assertEqual(target_failures({'groups':groups}),[])
        groups['uk_formal']['missed_or_abstained_concerns']=2
        self.assertEqual(len(target_failures({'groups':groups})),1)

    def setUp(self):
        self.rows=[dict(case_id='a',language='ro',gold=1,pair_group='pair'),dict(case_id='b',language='ro',gold=0,pair_group='pair')]

    def test_abstentions_cannot_hide_errors(self):
        result=evaluate(self.rows,[dict(case_id='a',predicted=None),dict(case_id='b',predicted=0)])
        m=result['groups']['ro_all']
        self.assertEqual(m['decided_accuracy'],1)
        self.assertEqual(m['strict_accuracy'],.5)
        self.assertEqual(m['coverage'],.5)
        self.assertEqual(m['concern_recall_including_abstentions'],0)
        self.assertEqual(result['paired_scenarios']['fully_correct'],0)

    def test_identity_errors(self):
        for preds in ([],[dict(case_id='x',predicted=1)], [dict(case_id='a',predicted=1)]*2):
            with self.assertRaises(ValueError):evaluate(self.rows,preds)

    def test_invalid_prediction(self):
        with self.assertRaises(ValueError):evaluate(self.rows,[dict(case_id='a',predicted=True)])

    def test_invariance_requires_correctness_as_well_as_consistency(self):
        rows=[dict(case_id=str(i),language='uk',gold=1,invariant_group='same') for i in range(2)]
        r=evaluate(rows,[dict(case_id=str(i),predicted=0) for i in range(2)])
        self.assertEqual(r['invariance'],dict(groups=1,inconsistent=0,fully_correct=0))

    def test_contradictory_invariance_labels_rejected(self):
        rows=[dict(r,invariant_group='bad') for r in self.rows]
        with self.assertRaises(ValueError):evaluate(rows,[dict(case_id='a',predicted=1),dict(case_id='b',predicted=0)])

    def test_triage_boundaries_and_no_authority(self):
        for score, expected in [(0,'no_signal'),(.2,'clarify'),(.7,'review'),(1,'review')]:
            r=triage(score,low=.2,high=.7,language='ro')
            self.assertEqual(r['decision'],expected)
            self.assertFalse(r['permits_external_action'])
            self.assertFalse(r['establishes_safety'])

    def test_unsupported_or_invalid_output_never_clears(self):
        for kw in [dict(score=0,language='en'),dict(score=0,language='ro',valid=False),dict(score=None,language='uk')]:
            self.assertEqual(triage(low=.2,high=.7,**kw)['decision'],'clarify')

    def test_bad_thresholds_and_scores(self):
        for score,low,high in [(float('nan'),.2,.8),(0,.8,.2),(True,.2,.8),(0,0,0)]:
            with self.assertRaises(ValueError):triage(score,low=low,high=high,language='ro')
