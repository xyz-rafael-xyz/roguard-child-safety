"""Corruption and scope tests for the reference completion audit."""
from copy import deepcopy
import json
import unittest
try:
    import numpy as np
except ImportError:
    np=None

from eval.verify_role_diversity_runtime_v10 import (check_observations,compare_outputs,
    finite_error,require_complete,verify_log)


class RoleReferenceAuditTests(unittest.TestCase):
    def report(self):
        return dict(complete=True,decision='REFERENCE_EXTRACTION_ONLY_REQUIRES_INDEPENDENT_AUDIT',
            training_performed=False,held_out_test_opened=False,qualified=False,
            languages={lang:dict(complete=True,modes={m:dict(complete=True) for m in ('v9','selected95')}) for lang in ('ro','uk')})

    def test_incomplete_and_scope_claims_fail(self):
        require_complete(self.report())
        for key,value in [('complete',False),('qualified',True),('training_performed',True),
                          ('held_out_test_opened',True),('decision','PASS')]:
            r=self.report();r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):require_complete(r)
        r=self.report();r['languages']['uk']['modes']['v9']['complete']=False
        with self.assertRaises(ValueError):require_complete(r)

    def test_invalid_numeric_observations_fail(self):
        for value in (float('nan'),float('inf'),-.1,.01,False,'0'):
            with self.subTest(value=value),self.assertRaises(ValueError):finite_error(value,.001)

    def log_fixture(self):
        events=[]
        for lang,dev in [('ro',4202),('uk',4635)]:
            for mode,stage,n in [('v9','train',1584),('selected95','train',1584),('selected95','dev',dev)]:
                positions=list(range(100,n+1,100));positions.append(n)
                for i in positions:
                    events.append(dict(language=lang,mode=mode,stage=stage,checked=i,rows=n,seconds=float(len(events)+1)))
        report=dict(seconds=200.)
        completion=dict(complete=True,decision='REFERENCE_EXTRACTION_ONLY_REQUIRES_INDEPENDENT_AUDIT',seconds=200.)
        return events,completion,report

    def test_full_progress_log_and_missing_events(self):
        events,last,report=self.log_fixture()
        render=lambda xs:'\n'.join(json.dumps(x) for x in xs)
        self.assertEqual(verify_log(render(events+[last]),report),154)
        for items in (events[:-1]+[last],list(reversed(events))+[last],events+[last,last],events):
            with self.subTest(length=len(items)),self.assertRaises(ValueError):verify_log(render(items),report)

    def test_log_invalid_time_and_terminal_report_mismatch(self):
        events,last,report=self.log_fixture()
        for value in (-1.,float('nan'),True):
            changed=deepcopy(events);changed[1]['seconds']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                verify_log('\n'.join(json.dumps(x) for x in changed+[last]),report)
        with self.assertRaises(ValueError):
            verify_log('\n'.join(json.dumps(x) for x in events+[last]),dict(seconds=201.))

    @unittest.skipIf(np is None,'NumPy array audit')
    def test_output_dtype_shape_and_tolerance(self):
        z=np.zeros((2,19),dtype=np.float32);spec=dict(mode='binary',threshold=.5)
        self.assertEqual(compare_outputs(z,z,spec,limit=1e-5,exact=True)['maximum_logit_error'],0.)
        for bad in (z.astype(np.float64),z[:1],z+1,np.full_like(z,np.nan)):
            with self.subTest(shape=bad.shape,dtype=bad.dtype),self.assertRaises(ValueError):
                compare_outputs(bad,z,spec,limit=1e-5,exact=True)

    @unittest.skipIf(np is None,'NumPy parity audit')
    def test_small_argmax_or_decision_change_rejected_for_dev(self):
        z=np.zeros((2,19),dtype=np.float32);actual=z.copy();actual[0,1]=1e-6;actual[1,-2]=1e-6
        spec=dict(mode='binary',threshold=.5)
        result=compare_outputs(actual,z,spec,limit=1e-5,exact=False)
        self.assertEqual(result['argmax_rows_differing'],1)
        self.assertEqual(result['fixed_threshold_decision_differences'],1)
        with self.assertRaisesRegex(ValueError,'parity'):compare_outputs(actual,z,spec,limit=1e-5,exact=True)

    def population(self):
        rows=[dict(case_id=str(i),partition='train',token_sha256=str(i),factors=dict(scope='direct',
              assertion='AFFIRMED',concern=int(i==2),bare=True)) for i in range(3)]
        obs=[dict(index=i,case_id=str(i),token_sha256=str(i),native_replay_error=.0002,
                  runtime_score_error=None if i==1 else 0.) for i in range(3)]
        record=dict(complete=True,train_checked=3,dev_checked=0,maximum_native_replay_error=.0002,
                    maximum_runtime_score_error=0.,runtime_groups_checked=2)
        return rows,obs,np.ones((3,2560),dtype=np.float32),np.zeros((3,19),dtype=np.float32),record

    @unittest.skipIf(np is None,'NumPy observation audit')
    def test_actual_observation_recount(self):
        rows,obs,x,z,record=self.population();r=check_observations(rows,obs,x,z,record,'v9')
        self.assertEqual(r['runtime_groups_checked'],2);self.assertEqual(r['train_native_observation_maximum'],.0002)

    @unittest.skipIf(np is None,'NumPy observation audit')
    def test_observation_id_order_token_and_probe_corruption(self):
        for key,value in [('index',1),('case_id','other'),('token_sha256','other'),
                          ('native_replay_error',.1),('runtime_score_error',None)]:
            rows,obs,x,z,record=self.population();obs[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):check_observations(rows,obs,x,z,record,'v9')
        rows,obs,x,z,record=self.population();obs[1]['runtime_score_error']=0.
        with self.assertRaises(ValueError):check_observations(rows,obs,x,z,record,'v9')

    @unittest.skipIf(np is None,'NumPy scope audit')
    def test_selected_dev_native_maximum_is_reported_not_independently_recounted(self):
        rows,obs,x,z,record=self.population();record['maximum_native_replay_error']=.0003
        result=check_observations(rows,obs,x,z,record,'selected95')
        self.assertFalse(result['dev_native_maximum_independently_recounted'])
        with self.assertRaises(ValueError):check_observations(rows,obs,x,z,record,'v9')
        record['maximum_native_replay_error']=.0001
        with self.assertRaises(ValueError):check_observations(rows,obs,x,z,record,'selected95')

    @unittest.skipIf(np is None,'NumPy report audit')
    def test_report_counts_and_array_values_fail(self):
        for key,value in [('train_checked',2),('runtime_groups_checked',3),('complete',False),
                          ('maximum_runtime_score_error',False),('dev_checked',False)]:
            rows,obs,x,z,record=self.population();record[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):check_observations(rows,obs,x,z,record,'v9')
        rows,obs,x,z,record=self.population();x[0,0]=np.inf
        with self.assertRaises(ValueError):check_observations(rows,obs,x,z,record,'v9')


if __name__=='__main__':unittest.main()
