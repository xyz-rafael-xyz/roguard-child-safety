import asyncio,json,tempfile,unittest
from pathlib import Path
from roguard.flow import FlowLedger,ToolBinding
from roguard.repair import repair_proposal
from roguard.uncertainty import CalibratedDecisions
from roguard.observability import Monitor
from roguard.long_context import analyze_long_conversation

class CoreTests(unittest.TestCase):
    def test_repair_preserves_goals_and_live_revoke(self):
        with FlowLedger() as l:
            for name in ('private','summary'):l.add_source(name,'fictional')
            l.permission('summary','lesson','teacher','export','grant')
            candidates=[dict(artifact_id='private',covers=['goal'],cost=8),dict(artifact_id='summary',covers=['goal'],cost=1)]
            args=dict(proposed=['private'],candidates=candidates,requirements=['goal'],purpose='lesson',recipient='teacher',operation='export')
            self.assertEqual(repair_proposal(l,**args)['selected'],['summary'])
            l.permission('summary','lesson','teacher','export','revoke')
            self.assertEqual(repair_proposal(l,**args)['status'],'NO_PERMITTED_REPAIR')
    def test_conformal_rank_and_identity(self):
        c=CalibratedDecisions.fit([.05,.15,.8,.9],[0,0,1,1],model_id='fixed',language='ro',alpha=.5)
        self.assertEqual(c.predict(.1,model_id='fixed',language='ro')['prediction_set'],[0])
        self.assertEqual(c.predict(.5,model_id='fixed',language='ro')['prediction_set'],[])
        with self.assertRaises(ValueError):c.predict(.1,model_id='changed',language='ro')
        small=CalibratedDecisions.fit([.1,.9],[0,1],model_id='x',language='uk')
        self.assertEqual(small.predict(.5,model_id='x',language='uk')['prediction_set'],[0,1])
    def test_monitor_isolation_retention_and_shift(self):
        clock=[100];m=Monitor(retention_seconds=60,max_events=100,clock=lambda:clock[0])
        def add(tenant,decision):m.record(tenant=tenant,language='ro',decision=decision,latency_ms=2,model_id='fixed',policy_version='v6')
        for _ in range(30):add('a','ALLOW')
        baseline=m.report('a');add('b','BLOCK');self.assertEqual(m.report('b')['languages']['ro']['count'],1)
        m.erase('a')
        for _ in range(30):add('a','BLOCK')
        self.assertTrue(m.report('a',baseline=baseline)['languages']['ro']['distribution_alert'])
        with self.assertRaises(TypeError):m.record(text='private')
        clock[0]=161;self.assertEqual(m.report('a')['languages']['ro']['count'],0);m.close()
    def test_long_context_retains_every_character(self):
        class Scorer:
            def scores(self,texts,language):return [.1]*len(texts)
            def thresholds(self,language):return dict(cutoff=.5,abstain_margin=.08)
        text='Eu am o carte și vreau să citesc. '*120
        report=analyze_long_conversation(dict(language='ro',turns=[dict(role='minor',text=text)]),Scorer())
        seen=set()
        for window in report['windows']:
            for ref in window['references']:seen.update(range(ref['start'],ref['end']))
        self.assertEqual(len(seen),len(text));self.assertEqual(report['decision'],'INSUFFICIENT_EVIDENCE');self.assertNotIn(text,json.dumps(report))

try:
    from cryptography.fernet import Fernet
except ImportError:Fernet=None

@unittest.skipIf(Fernet is None,'gateway extra not installed')
class GatewayTests(unittest.TestCase):
    def setUp(self):
        from roguard.durable import DurableGateway
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'state.db';self.key=Fernet.generate_key();self.clock=[100];self.calls=[]
        self.bindings={'a':{'send':ToolBinding('lesson','teacher','export',lambda values:self.calls.append(values))}}
        self.g=DurableGateway(self.path,self.key,self.bindings,binding_version='v1',clock=lambda:self.clock[0]);self.g.create('a','s');self.g.mutate('a','s','source',dict(artifact_id='item',value='FICTIONAL_SECRET'))
        self.g.mutate('a','s','permission',dict(root='item',purpose='lesson',recipient='teacher',operation='export',action='grant',expires_at=150))
    def tearDown(self):self.tmp.cleanup()
    def job(self):
        j=self.g.enqueue('a','s','send',['item'])['job_id'];self.g.approve('a',j);return j
    def test_encryption_restart_and_single_attempt(self):
        from roguard.durable import DurableGateway
        j=self.job();self.assertNotIn(b'FICTIONAL_SECRET',self.path.read_bytes())
        other=DurableGateway(self.path,self.key,self.bindings,binding_version='v1',clock=lambda:self.clock[0])
        self.assertEqual(other.execute('a',j)['status'],'EXECUTED');other.execute('a',j);self.assertEqual(len(self.calls),1)
    def test_tenant_and_binding_isolation(self):
        from roguard.durable import DurableGateway
        j=self.job()
        with self.assertRaises(KeyError):self.g.status('b',j)
        other=DurableGateway(self.path,self.key,self.bindings,binding_version='v2',clock=lambda:self.clock[0])
        self.assertEqual(other.execute('a',j)['status'],'BLOCK');self.assertEqual(self.calls,[])
    def test_revoke_expiry_cancel_and_clock_rollback(self):
        j=self.job();self.g.mutate('a','s','permission',dict(root='item',purpose='lesson',recipient='teacher',operation='export',action='revoke'))
        self.assertEqual(self.g.execute('a',j)['status'],'STALE');self.assertEqual(self.calls,[])
        self.g.mutate('a','s','permission',dict(root='item',purpose='lesson',recipient='teacher',operation='export',action='grant',expires_at=150))
        j=self.job();self.g.cancel('a',j);self.assertEqual(self.g.execute('a',j)['status'],'CANCELLED')
        j=self.job();self.clock[0]=151;self.assertEqual(self.g.execute('a',j)['status'],'BLOCK')
        self.clock[0]=100
        with self.assertRaises(ValueError):self.g.enqueue('a','s','send',['item'])
    def test_crash_marker_never_retried(self):
        j=self.job()
        with self.g._tx() as db:db.execute("UPDATE jobs SET status='STARTED' WHERE id=?",(j,))
        self.assertEqual(self.g.execute('a',j)['status'],'EXECUTION_UNCERTAIN');self.assertEqual(self.calls,[])
    def test_callback_failure_never_retried(self):
        def fail(values):self.calls.append(1);raise RuntimeError('PRIVATE')
        self.g.bindings['a']['send']=ToolBinding('lesson','teacher','export',fail)
        j=self.job();r=self.g.execute('a',j);self.assertEqual(r['status'],'EXECUTION_UNCERTAIN');self.g.execute('a',j);self.assertEqual(len(self.calls),1);self.assertNotIn('PRIVATE',json.dumps(r))
    def test_service_auth_roles_and_body(self):
        from roguard.service import GatewayApp
        app=GatewayApp(self.g,{'a'*32:dict(tenant='a',role='reader'),'b'*32:dict(tenant='b',role='admin')})
        async def request(path,body,token='a'*32):
            sent=[];messages=[dict(type='http.request',body=json.dumps(body).encode())]
            async def receive():return messages.pop(0)
            async def send(x):sent.append(x)
            await app(dict(type='http',method='POST',path=path,headers=[(b'authorization',('Bearer '+token).encode()),(b'content-type',b'application/json')]),receive,send)
            return sent[0]['status'],json.loads(sent[1]['body'])
        self.assertEqual(asyncio.run(request('/v1/session/create',dict(session='x')))[0],403)
        self.assertEqual(asyncio.run(request('/v1/job/status',dict(job=self.job()),'b'*32))[0],400)
        self.assertEqual(asyncio.run(request('/v1/analyze',{},'invalid'))[0],401)

if __name__=='__main__':unittest.main()

class AdditionalReliabilityTests(unittest.TestCase):
    def test_feature_support_detects_distance_and_identity(self):
        from roguard.uncertainty import FeatureSupport
        s=FeatureSupport.fit([[0.,0.],[1.,1.],[.5,.5]],[[.2,.2],[.8,.8]],model_id='a')
        self.assertFalse(s.check([.5,.5],model_id='a')['outside_reference_support'])
        self.assertTrue(s.check([20.,20.],model_id='a')['outside_reference_support'])
        with self.assertRaises(ValueError):s.check([0.,0.],model_id='b')
    def test_remote_target_requires_explicit_secure_destination(self):
        from roguard.targets import ChatTarget,parse_reply
        for url in ('http://example.com/api','https://user:password@example.com','file:///tmp/test'):
            with self.assertRaises(ValueError):ChatTarget(url,'test')
        with self.assertRaises(ValueError):parse_reply('```json\n{}\n```')
    def test_otel_export_has_only_declared_metadata(self):
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        except ImportError:self.skipTest('observability extra not installed')
        exporter=InMemorySpanExporter();provider=TracerProvider();provider.add_span_processor(SimpleSpanProcessor(exporter))
        m=Monitor(tracer=provider.get_tracer('roguard-test'));m.record(tenant='a',language='uk',decision='BLOCK',latency_ms=1,model_id='fixed',policy_version='v6')
        span=exporter.get_finished_spans()[0]
        self.assertEqual(set(span.attributes),{'roguard.language','roguard.decision','roguard.latency_ms','roguard.model_id','roguard.policy_version','roguard.dataset_id'})
        self.assertEqual(span.events,());m.close();provider.shutdown()
