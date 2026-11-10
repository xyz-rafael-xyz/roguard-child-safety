"""Reference alignment, model identity and early rejection checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np=None

from roguard.trajectory_data import file_sha
from training.role_diversity_runtime_v10 import (align_development,check_bindings,
    load_registered,validate_array,validate_specs)


class RoleRuntimeTests(unittest.TestCase):
    def fixture(self):
        return [dict(case_id=str(i),language='ro',partition='dev',fields=[0]*6,
                     gold=0,group='fixture',token_sha256=str(i),payload={'id':i}) for i in range(2)]

    def test_preserves_reference_order_instead_of_recovered_order(self):
        rows=self.fixture();metadata=list(reversed(deepcopy(rows)))
        actual=align_development(rows,metadata,'ro')
        self.assertEqual([r['case_id'] for r in actual],['1','0'])

    def test_duplicate_or_missing_reference_fails(self):
        rows=self.fixture()
        for metadata in (rows[:-1],[rows[0],rows[0]]):
            with self.subTest(metadata=metadata),self.assertRaises(ValueError):
                align_development(rows,metadata,'ro')

    def test_gold_field_token_or_partition_changes_fail(self):
        for key,value in [('gold',1),('fields',[1]*6),('token_sha256','other'),('partition','train')]:
            rows=self.fixture();metadata=deepcopy(rows);metadata[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):align_development(rows,metadata,'ro')

    def test_duplicate_recovered_id_fails(self):
        rows=self.fixture();rows[1]['case_id']=rows[0]['case_id']
        with self.assertRaises(ValueError):align_development(rows,rows,'ro')

    def test_model_identity_and_policy_are_bound(self):
        original=dict(adapter_sha256='v9',mode='policy',threshold=.5,language='ro',policy='fixed')
        initial=dict(original,adapter_sha256='new',mode='binary',threshold=.6)
        validate_specs(original,initial,'v9','new')
        for change in ({'policy':'changed'},{'language':'uk'},{'adapter_sha256':'other'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_specs(original,dict(initial,**change),'v9','new')

    def test_bound_source_and_private_input_mutations_fail(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'source').write_text('a');(p/'input').write_text('b')
            plan=dict(public={'source':file_sha(p/'source')},private={'input':file_sha(p/'input')})
            check_bindings(p,p,plan)
            for name in ('source','input'):
                before=(p/name).read_text();(p/name).write_text('changed')
                with self.subTest(name=name),self.assertRaises(ValueError):check_bindings(p,p,plan)
                (p/name).write_text(before)

    @unittest.skipIf(np is None,'NumPy numerical check')
    def test_dtype_shape_and_nonfinite_reference_rejection(self):
        validate_array(np.zeros((2,19),dtype=np.float32),(2,19))
        for array in (np.zeros((2,19),dtype=np.float64),np.zeros((1,19),dtype=np.float32),
                      np.full((2,19),np.nan,dtype=np.float32),np.full((2,19),np.inf,dtype=np.float32)):
            with self.subTest(shape=array.shape,dtype=array.dtype),self.assertRaises(ValueError):
                validate_array(array,(2,19))

    @unittest.skipIf(np is None,'NumPy preflight check')
    def test_changed_plan_hash_rejected_before_loading_assets(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);plan=p/'plan.json';plan.write_text('{}')
            with self.assertRaisesRegex(ValueError,'plan differs'):
                load_registered(p,p,plan,'0'*64)

    @unittest.skipIf(np is None,'NumPy preflight check')
    def test_population_or_tolerance_changes_fail_before_encoder(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);file=p/'plan.json'
            base=dict(format='roguard-role-diversity-runtime-plan-1',rows=3168,development_rows=8837,
                      native_replay_tolerance=1e-3,reference_logit_tolerance=1e-5,
                      runtime_score_tolerance=1e-10,training_performed=False)
            for key,value in [('rows',3167),('development_rows',1),('native_replay_tolerance',.1),
                              ('reference_logit_tolerance',.01),('runtime_score_tolerance',.01),
                              ('training_performed',True)]:
                file.write_text(json.dumps(dict(base,**{key:value})))
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Population, tolerances'):
                    load_registered(p,p,file,file_sha(file))


if __name__=='__main__':unittest.main()
