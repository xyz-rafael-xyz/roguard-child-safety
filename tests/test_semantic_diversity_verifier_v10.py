"""Mutation tests for the independently implemented corpus verifier."""
from copy import deepcopy
import unittest

from roguard.assurance import digest
from roguard.gemma4_contract import PREFIX_TOKENS
from training.semantic_diversity_v10 import build
from eval.verify_semantic_diversity_v10 import verify_population


def tokens(payload):
    # Unique bounded fixture tokens; the actual audit uses the runtime tokenizer.
    return PREFIX_TOKENS+list(bytes.fromhex(digest(payload)))


class SemanticDiversityVerifierTests(unittest.TestCase):
    def setUp(self):
        self.manifest=dict(format='roguard-authored-semantic-diversity-1',partition='train',
            synthetic_only=True,independently_annotated=False,pairs=[dict(family='fixture',
                ro=['Îmi place desenul.','Plâng de frică.'],uk=['Мені подобається малювати.','Я плачу від страху.'])])
        self.rows=build(self.manifest,[])
        for r in self.rows:
            t=tokens(r['payload']);r.update(tokens=t,token_count=len(t),token_sha256=digest(t))

    def test_complete_recount(self):
        r=verify_population(self.rows,self.manifest,[],tokens)
        self.assertEqual(r['rows'],132)
        self.assertEqual(sum(c['rows'] for c in r['counts']),132)

    def test_missing_and_duplicate_cases_fail(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]]):
            with self.subTest(n=len(rows)),self.assertRaises(ValueError):
                verify_population(rows,self.manifest,[],tokens)

    def test_label_or_provenance_corruption_fails(self):
        for key,value in [('gold',1),('fields',[0,0,0,0,0,1]),('partition','dev'),
                          ('independently_annotated',True),('case_id','changed')]:
            rows=deepcopy(self.rows);rows[0][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'content, provenance or label'):
                verify_population(rows,self.manifest,[],tokens)

    def test_payload_and_prior_retraction_corruption_fail(self):
        for position in (0,1):
            rows=deepcopy(self.rows)
            row=next(r for r in rows if r['attributes']['assertion']=='RETRACTED')
            row['payload']['turns'][position]['text']='different'
            with self.subTest(position=position),self.assertRaises(ValueError):
                verify_population(rows,self.manifest,[],tokens)

    def test_token_corruption_fails_even_with_updated_digest(self):
        rows=deepcopy(self.rows);rows[0]['tokens'][-1]+=1
        rows[0]['token_sha256']=digest(rows[0]['tokens'])
        with self.assertRaisesRegex(ValueError,'token replay'):
            verify_population(rows,self.manifest,[],tokens)

    def test_protected_existing_core_fails(self):
        r=self.rows[0]
        protected=[dict(language='ro',partition='dev',payload=r['payload'],token_sha256=r['token_sha256'])]
        with self.assertRaisesRegex(ValueError,'authored core'):
            verify_population(self.rows,self.manifest,protected,tokens)

    def test_reserved_test_partition_fails(self):
        r=self.rows[0]
        protected=[dict(language='ro',partition='test',payload=r['payload'],token_sha256=r['token_sha256'])]
        with self.assertRaisesRegex(ValueError,'Reserved partition'):
            verify_population(self.rows,self.manifest,protected,tokens)

    def test_repeated_family_manifest_fails(self):
        self.manifest['pairs'].append(deepcopy(self.manifest['pairs'][0]))
        with self.assertRaisesRegex(ValueError,'family inventory'):
            verify_population(self.rows,self.manifest,[],tokens)


if __name__=='__main__':unittest.main()
