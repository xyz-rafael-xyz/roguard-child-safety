"""Check corpus leakage gates, role semantics and conservative policy targets."""
from copy import deepcopy
import unittest

from roguard.assurance import digest
from training.semantic_diversity_v10 import build, summarize, validate_manifest


def manifest():
    return dict(format='roguard-authored-semantic-diversity-1', partition='train',
                synthetic_only=True, independently_annotated=False,
                pairs=[dict(family='fixture', ro=['Îmi place desenul.', 'Plâng de frică.'],
                            uk=['Мені подобається малювати.', 'Я плачу від страху.'])])


class SemanticDiversityTests(unittest.TestCase):
    def test_policy_targets_and_complete_population(self):
        rows = build(manifest(), [])
        self.assertEqual(len(rows), 132)
        self.assertEqual(len({digest(r['payload']) for r in rows}), 132)
        self.assertEqual(summarize(rows)['distinct_cores'], {'ro':2, 'uk':2})
        for r in rows:
            a=r['attributes']; f=r['factors']
            # Independent explicit truth table for the limited generated worlds.
            expected = int(a['concern']=='1' and (a['assertion']=='RETRACTED'
                           or (f['scope']=='direct' and a['assertion']!='NEGATED')))
            self.assertEqual(r['gold'], expected)
            self.assertEqual(r['partition'], 'train')
            self.assertFalse(r['independently_annotated'])
            if f['scope']=='fiction':
                self.assertEqual((a['subject'],a['source']),('FICTIONAL','FICTION'))
            else:
                self.assertEqual((a['subject'],a['source']),('SELF','DIRECT'))

    def test_retraction_has_real_prior_statement_in_input(self):
        for r in build(manifest(), []):
            turns=r['payload']['turns']
            if r['attributes']['assertion']=='RETRACTED':
                self.assertEqual(turns[0],dict(role='minor',text=r['core']))
                self.assertEqual(r['attributes']['relation'],'RETRACTS')
                self.assertIn('«'+r['core']+'»',turns[-1]['text'])
            else:
                self.assertEqual(r['attributes']['relation'],'NEW')

    def test_protected_core_rejected_even_inside_longer_focus(self):
        p=dict(partition='dev',payload=dict(language='ro',turns=[
            dict(role='minor',text='Text anterior: «ÎMI PLACE DESENUL.»')]))
        with self.assertRaisesRegex(ValueError,'Authored core occurs'):
            build(manifest(),[p])

    def test_reserved_partition_is_never_accepted(self):
        p=dict(partition='test',payload=dict(language='ro',turns=[dict(role='minor',text='Alt text.')]))
        with self.assertRaisesRegex(ValueError,'already exposed'):
            build(manifest(),[p])

    def test_missing_duplicate_and_quoted_cores_fail(self):
        mutations=[]
        m=manifest();m['pairs'][0]['uk']=[];mutations.append(m)
        m=manifest();m['pairs'][0]['ro'][1]=m['pairs'][0]['ro'][0];mutations.append(m)
        m=manifest();m['pairs'].append(deepcopy(m['pairs'][0]));mutations.append(m)
        m=manifest();m['pairs'][0]['ro'][0]='«Text»';mutations.append(m)
        for m in mutations:
            with self.subTest(m=m),self.assertRaises(ValueError):validate_manifest(m)

    def test_false_independent_claim_and_partition_change_rejected(self):
        for key,value in [('partition','dev'),('independently_annotated',True),('synthetic_only',False)]:
            m=manifest();m[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):build(m,[])

    def test_generation_is_reproducible_without_mutating_manifest(self):
        m=manifest(); original=deepcopy(m)
        self.assertEqual(build(m,[]),build(m,[]))
        self.assertEqual(m,original)


if __name__=='__main__':
    unittest.main()
