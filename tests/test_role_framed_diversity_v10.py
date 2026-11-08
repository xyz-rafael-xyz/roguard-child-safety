"""Verify repaired role claims, conserved labels and mutation rejection."""
from copy import deepcopy
import unittest
from roguard.assurance import digest
from roguard.gemma4_contract import PREFIX_TOKENS
from training.semantic_diversity_v10 import build
from training.role_framed_diversity_v10 import continue_row,FRAMES
from eval.verify_role_framed_diversity_v10 import verify_rows


def tokenize(payload):return PREFIX_TOKENS+list(bytes.fromhex(digest(payload)))


class RoleFramedDiversityTests(unittest.TestCase):
    def setUp(self):
        manifest=dict(format='roguard-authored-semantic-diversity-1',partition='train',
            synthetic_only=True,independently_annotated=False,pairs=[dict(family='fixture',
            ro=['Îmi place desenul.','Plâng de frică.'],uk=['Мені подобається малювати.','Я плачу від страху.'])])
        self.parents=build(manifest,[])
        for r in self.parents:
            t=tokenize(r['payload']);r.update(tokens=t,token_count=len(t),token_sha256=digest(t))
        self.original=deepcopy(self.parents)
        self.rows=[continue_row(r,tokenize) for r in self.parents]

    def test_complete_population_labels_and_nonmutation(self):
        result=verify_rows(self.parents,self.rows,tokenize,[])
        self.assertEqual((result['rows'],result['fiction_rows_changed'],result['direct_rows_unchanged']),(132,64,68))
        self.assertEqual(self.parents,self.original)
        for p,r in zip(self.parents,self.rows):
            self.assertEqual((p['fields'],p['gold'],p['core']),(r['fields'],r['gold'],r['core']))

    def test_minor_explicit_fiction_and_prior_retraction(self):
        for p,r in zip(self.parents,self.rows):
            if r['framing_changed']:
                frame=FRAMES[r['language']];turns=r['payload']['turns']
                self.assertTrue(turns[-1]['text'].startswith(frame['focus']))
                self.assertEqual(turns[-1]['role'],'minor')
                if r['attributes']['assertion']=='RETRACTED':
                    self.assertIn('«'+r['core']+'»',turns[0]['text'])
                    if r['attributes']['concern']=='1':self.assertEqual(r['gold'],1)
            else:self.assertEqual(r['payload'],p['payload'])

    def test_unframed_fiction_and_changed_direct_fail(self):
        for fiction in (True,False):
            rows=deepcopy(self.rows);i=next(i for i,r in enumerate(rows) if r['framing_changed']==fiction)
            rows[i]['payload']=deepcopy(self.parents[i]['payload'])
            if not fiction:rows[i]['payload']['turns'][-1]['text']+=' different'
            with self.subTest(fiction=fiction),self.assertRaises(ValueError):
                verify_rows(self.parents,rows,tokenize,[])

    def test_label_core_factor_and_provenance_corruption_fail(self):
        for key,value in [('gold',1),('core','different'),('fields',[2,0,0,2,0,1]),
                          ('parent_row_sha256','0'*64),('framing_changed',True),('extra',True)]:
            rows=deepcopy(self.rows);rows[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):verify_rows(self.parents,rows,tokenize,[])

    def test_duplicate_missing_and_protected_tokens_fail(self):
        for rows in (self.rows[:-1],self.rows[:-1]+[self.rows[0]]):
            with self.subTest(n=len(rows)),self.assertRaises(ValueError):verify_rows(self.parents,rows,tokenize,[])
        p=deepcopy(self.rows[0]);p['partition']='dev'
        with self.assertRaises(ValueError):verify_rows(self.parents,self.rows,tokenize,[p])

    def test_changed_tokens_with_forged_hash_fail(self):
        rows=deepcopy(self.rows);rows[0]['tokens'][-1]+=1;rows[0]['token_sha256']=digest(rows[0]['tokens'])
        with self.assertRaisesRegex(ValueError,'tokens differ'):verify_rows(self.parents,rows,tokenize,[])

    def test_transform_rejects_unregistered_parent(self):
        for key,value in [('partition','dev'),('case_id','unknown'),('independently_annotated',True)]:
            p=deepcopy(self.parents[0]);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):continue_row(p,tokenize)


if __name__=='__main__':unittest.main()
