"""Provenance, live revocation, approval replay and minimum disclosure."""
import itertools
import random
import threading
import unittest
from roguard.flow import FlowLedger, ToolBinding, ToolGateway
from roguard.disclosure import minimize_disclosure

CTX = dict(purpose='tutoring', recipient='student', operation='use')


def grant(ledger, root='a', action='grant', **kwargs):
    ledger.permission(root, action=action, **CTX, **kwargs)


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.ledger = FlowLedger(); self.ledger.add_source('a', 'Fictional source A.')
        self.addCleanup(self.ledger.close)

    def test_derived_union_live_revocation_and_unknown_sources(self):
        l=self.ledger; l.add_source('b','Fictional source B.'); grant(l)
        l.derive('summary','A summary.',['a']);l.derive('combined','Combined.',['summary','b'])
        self.assertEqual(l.check(['summary'],**CTX)['decision'],'ALLOW')
        self.assertEqual(l.check(['combined'],**CTX)['decision'],'BLOCK')
        grant(l,'b'); self.assertEqual(l.check(['combined'],**CTX)['decision'],'ALLOW')
        grant(l,action='revoke'); self.assertEqual(l.check(['combined'],**CTX)['decision'],'BLOCK')
        self.assertEqual(l.describe('combined')['sources'],['a','b'])
        with self.assertRaises(ValueError):grant(l,'summary')
        self.assertEqual(l.check(['absent'],**CTX)['decision'],'BLOCK')

    def test_incomplete_provenance_cannot_be_repaired_by_derivation_flag(self):
        l=self.ledger;l.add_source('b','Unknown origin.',complete=False);grant(l,'b')
        l.derive('c','Copy.',['b'],complete=True)
        self.assertIn('INCOMPLETE_PROVENANCE',l.check(['c'],**CTX)['reason_codes'])
        with self.assertRaises(ValueError):l.derive('d','Copy.',[])

    def test_expiry_pause_resume_and_no_resurrection(self):
        l=self.ledger;grant(l,expires_at=5);l.advance(4)
        self.assertEqual(l.check(['a'],**CTX)['decision'],'ALLOW')
        grant(l,action='pause');self.assertEqual(l.check(['a'],**CTX)['decision'],'BLOCK')
        grant(l,action='resume');l.advance(5)
        self.assertEqual(l.check(['a'],**CTX)['decision'],'BLOCK')
        grant(l,action='revoke');grant(l,action='resume')
        self.assertEqual(l.check(['a'],**CTX)['decision'],'BLOCK')
        with self.assertRaises(ValueError):l.advance(4)
        with self.assertRaises(ValueError):grant(l,expires_at=5)
        with self.assertRaises(ValueError):l.advance(True)

    def test_scope_binding_and_immutable_values(self):
        l=self.ledger;grant(l)
        for key,value in [('purpose','progress'),('recipient','guardian'),('operation','memory')]:
            context={**CTX,key:value};self.assertEqual(l.check(['a'],**context)['decision'],'BLOCK')
        with self.assertRaises(ValueError):l.add_source('a','Replace')
        l.close()
        with self.assertRaises(ValueError):l.check(['a'],**CTX)

    def test_gateway_requires_bound_single_use_current_host_token(self):
        l=self.ledger;grant(l);calls=[]
        gateway=ToolGateway(l,{'deliver':ToolBinding(**CTX,callback=lambda values:calls.append(values))})
        self.assertEqual(gateway.dispatch('deliver',['a'],execute=True)['decision'],'REVIEW')
        self.assertEqual(gateway.dispatch('deliver',['a'],approval=True,execute=True)['decision'],'REVIEW')
        token=gateway.approve('deliver',['a'])
        self.assertEqual(gateway.dispatch('deliver',['a'],approval=token)['decision'],'READY')
        self.assertFalse(calls)
        self.assertEqual(gateway.dispatch('deliver',['a'],approval=token,execute=True)['decision'],'EXECUTED')
        self.assertEqual(len(calls),1)
        self.assertEqual(gateway.dispatch('deliver',['a'],approval=token,execute=True)['decision'],'REVIEW')
        token=gateway.approve('deliver',['a']);grant(l,action='revoke')
        self.assertEqual(gateway.dispatch('deliver',['a'],approval=token,execute=True)['decision'],'BLOCK')
        grant(l)
        self.assertEqual(gateway.dispatch('deliver',['a'],approval=token,execute=True)['decision'],'REVIEW')
        self.assertEqual(len(calls),1)

    def test_external_binding_cannot_disable_approval(self):
        for operation in ['export','tool']:
            with self.assertRaises(ValueError):
                ToolBinding('task','recipient',operation,lambda values:None,requires_approval=False)

    def test_gateway_output_provenance_and_partial_failure_no_retry(self):
        l=self.ledger;grant(l)
        known=ToolGateway(l,{'summarize':ToolBinding(**CTX,callback=lambda v:'Short version.',output_provenance_complete=True)})
        token=known.approve('summarize',['a'])
        report=known.dispatch('summarize',['a'],approval=token,execute=True,output_id='result')
        self.assertEqual(report['decision'],'EXECUTED');self.assertEqual(l.describe('result')['sources'],['a'])
        grant(l,action='revoke');self.assertEqual(l.check(['result'],**CTX)['decision'],'BLOCK')
        grant(l)
        unknown=ToolGateway(l,{'external':ToolBinding(**CTX,callback=lambda v:'New information.')})
        unknown.dispatch('external',['a'],approval=unknown.approve('external',['a']),execute=True,output_id='external_result')
        self.assertEqual(l.check(['external_result'],**CTX)['decision'],'BLOCK')
        calls=[]
        def failing(values):calls.append(1);raise RuntimeError('DO NOT LEAK THIS')
        failing_gateway=ToolGateway(l,{'f':ToolBinding(**CTX,callback=failing)})
        token=failing_gateway.approve('f',['a'])
        report=failing_gateway.dispatch('f',['a'],approval=token,execute=True)
        self.assertEqual(report['decision'],'EXECUTION_UNCERTAIN');self.assertIsNone(report['external_action_taken'])
        self.assertNotIn('DO NOT LEAK',str(report));self.assertEqual(len(calls),1)
        self.assertEqual(failing_gateway.dispatch('f',['a'],approval=token,execute=True)['decision'],'REVIEW')

    def test_revocation_serializes_with_dispatch_and_blocks_next_action(self):
        l=self.ledger;grant(l);entered=threading.Event();release=threading.Event();revoked=threading.Event()
        def callback(values):entered.set();self.assertTrue(release.wait(2))
        gateway=ToolGateway(l,{'t':ToolBinding(**CTX,callback=callback)})
        token=gateway.approve('t',['a'])
        worker=threading.Thread(target=lambda:gateway.dispatch('t',['a'],approval=token,execute=True))
        worker.start();self.assertTrue(entered.wait(2))
        def revoke():grant(l,action='revoke');revoked.set()
        revoker=threading.Thread(target=revoke);revoker.start()
        self.assertFalse(revoked.is_set());release.set();worker.join(2);revoker.join(2)
        self.assertFalse(worker.is_alive());self.assertTrue(revoked.is_set())
        self.assertEqual(gateway.dispatch('t',['a'],approval=token,execute=True)['decision'],'BLOCK')


class DisclosureTests(unittest.TestCase):
    def test_optimizer_matches_independent_exhaustive_subsets(self):
        rng=random.Random(23)
        for _ in range(60):
            with FlowLedger() as ledger:
                candidates=[];permitted=set();requirements=['g0','g1','g2']
                for i in range(7):
                    name=f'a{i}';ledger.add_source(name,'Fictional field.')
                    if rng.choice([True,True,False]):grant(ledger,name);permitted.add(name)
                    candidates.append({'artifact_id':name,'covers':[g for g in requirements if rng.choice([True,False])],'cost':rng.randrange(5)})
                options=[]
                for size in range(8):
                    for subset in itertools.combinations(candidates,size):
                        names=tuple(sorted(c['artifact_id'] for c in subset))
                        covers=set().union(*(set(c['covers']) for c in subset))
                        if set(names)<=permitted and set(requirements)<=covers:
                            options.append((sum(c['cost'] for c in subset),len(names),names))
                result=minimize_disclosure(ledger,candidates,requirements,**CTX)
                if options:
                    best=min(options);self.assertEqual((result['declared_cost'],len(result['selected']),tuple(result['selected'])),best)
                else:self.assertEqual(result['status'],'INSUFFICIENT_PERMITTED_COVERAGE')
                self.assertFalse(result['execution_authorized']);self.assertFalse(result['semantic_sufficiency_verified'])

    def test_plan_is_not_authority_after_revocation(self):
        with FlowLedger() as ledger:
            ledger.add_source('a','Private fictional field.');grant(ledger)
            plan=minimize_disclosure(ledger,[{'artifact_id':'a','covers':['need'],'cost':1}],['need'],**CTX)
            self.assertEqual(plan['selected'],['a']);grant(ledger,action='revoke')
            self.assertEqual(ledger.check(plan['selected'],**CTX)['decision'],'BLOCK')

if __name__=='__main__':unittest.main()
