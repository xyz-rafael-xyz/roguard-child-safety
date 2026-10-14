"""Evidence/permission boundaries, model export behavior and context input limits."""
import copy
import json
from pathlib import Path
import unittest
from roguard.context import ContextModel, analyze_context, compare_context, validate
from roguard.context_session import ContextSession

class Stub:
    backend='test';identity='test-model'
    def predict(self,texts,language):
        return [{'supported':True,'heads':{'frame':{'label':'CORRECTION' if i else 'SELF_CURRENT','prediction_set':['CORRECTION' if i else 'SELF_CURRENT']},'signal':{'label':1,'prediction_set':[1]}}} for i,t in enumerate(texts)]

class ContextTests(unittest.TestCase):
    def test_references_are_exact_but_unverified_and_text_free(self):
        payload={'language':'ro','turns':[{'role':'minor','text':'Îmi este greu.'},{'role':'minor','text':'Corectez afirmația.'}]}
        result=analyze_context(payload,Stub())
        self.assertEqual(result['review_turn_indices'],[0,1])
        self.assertEqual(result['unresolved_turn_indices'],[1])
        self.assertEqual(result['evidence'][1]['context_turn_indices'],[0])
        self.assertEqual(result['evidence'][0]['candidate_reference']['end'],len(payload['turns'][0]['text']))
        self.assertFalse(result['evidence'][0]['verified'])
        self.assertFalse(result['authority_from_model'])
        self.assertNotIn(payload['turns'][0]['text'],json.dumps(result,ensure_ascii=False))

    def test_malformed_and_oversized_input_rejected(self):
        base={'language':'uk','turns':[{'role':'minor','text':'Приклад'}]}
        for payload in ({},dict(base,language='en'),dict(base,turns=[]),dict(base,turns=base['turns']*13),dict(base,turns=[{'role':'minor','text':'x'*2001}]),dict(base,turns=[{'role':'minor','text':'x'*2000}]*4),dict(base,turns=[{'role':[],'text':'x'}]),dict(base,authority=True)):
            with self.assertRaises(ValueError):validate(payload)

    def test_no_minor_is_insufficient_evidence(self):
        r=analyze_context({'language':'ro','turns':[{'role':'assistant','text':'Text'}]},Stub())
        self.assertEqual(r['decision'],'INSUFFICIENT_EVIDENCE')
        self.assertEqual(r['evidence'],[])

    def test_unsupported_context_is_visible(self):
        class Unsupported(Stub):
            def predict(self,texts,language):return [{'supported':False,'heads':{},'reason':'SEMANTIC_CONTEXT_LIMIT'} for t in texts]
        r=analyze_context({'language':'ro','turns':[{'role':'minor','text':'Text'}]},Unsupported())
        self.assertEqual(r['decision'],'REVIEW')
        self.assertFalse(r['evidence'][0]['supported'])
        self.assertEqual(r['unresolved_turn_indices'],[0])

    def test_model_change_cannot_be_compared(self):
        r=analyze_context({'language':'ro','turns':[{'role':'minor','text':'Text'}]},Stub())
        with self.assertRaises(ValueError):compare_context(r,dict(r,model_sha256='other'))
        self.assertEqual(compare_context(r,r)['new_review_turn_indices'],[])

    def test_no_grant_from_model_and_summary_retains_revocation(self):
        with ContextSession('ro',Stub()) as s:
            s.append('minor','4 + 3 = 7');s.append('minor','Un mesaj privat fictiv.')
            args=dict(purpose='tutoring',recipient='teacher')
            p=s.propose_summary([0,1],{'0':['learning'],'1':[]},['learning'],**args)
            self.assertEqual(p['proposal']['status'],'NO_PERMITTED_REPAIR')
            s.ledger.permission('turn_0','tutoring','teacher','export','grant')
            p=s.propose_summary([0,1],{'0':['learning'],'1':[]},['learning'],draft_id='summary',**args)
            self.assertEqual(p['proposal']['selected'],['turn_0'])
            self.assertFalse(p['execution_authorized'])
            self.assertEqual(s.ledger.check(['summary'],'tutoring','teacher','export')['decision'],'ALLOW')
            s.ledger.permission('turn_0','tutoring','teacher','export','revoke')
            self.assertEqual(s.ledger.check(['summary'],'tutoring','teacher','export')['decision'],'BLOCK')
            self.assertEqual(s.ledger.check(['summary'],'other','teacher','export')['decision'],'BLOCK')
        with self.assertRaises(ValueError):s.append('minor','later')

    def test_no_excerpt_declassification_and_coverage_required(self):
        with ContextSession('uk',Stub()) as s:
            s.append('minor','Вправа і приватна нотатка в одному повідомленні.')
            with self.assertRaises(ValueError):s.propose_summary([0],{},['learning'],purpose='tutoring',recipient='teacher')
            p=s.propose_summary([0],{'0':['learning']},['learning'],purpose='tutoring',recipient='teacher',draft_id='summary')
            self.assertIsNone(p['draft_artifact_id'])
            with self.assertRaises(ValueError):s.propose_summary([True],{'True':[]},[],purpose='tutoring',recipient='teacher')

    def test_packaged_character_model_is_finite_and_bounded(self):
        model=ContextModel('character')
        for language in ('ro','uk'):
            result=analyze_context(json.loads(Path('examples/context_'+language+'.json').read_text()),model)
            self.assertEqual(len(result['model_sha256']),64)
            self.assertFalse(result['external_action_taken'])
            for item in result['evidence']:
                self.assertTrue(set(item['frame_set'])<=set(model.spec['heads']['frame']['classes']))
                if item['frame'] is not None:self.assertEqual(item['frame_set'],[item['frame']])

if __name__=='__main__':unittest.main()

class ContextBoundaryTests(unittest.TestCase):
    def test_ukrainian_latin_only_is_explicitly_unsupported(self):
        result=analyze_context({'language':'uk','turns':[{'role':'minor','text':'Meni samotno'}]},ContextModel('character'))
        self.assertEqual(result['evidence'][0]['reason'],'UKRAINIAN_SCRIPT_REQUIRED')
        self.assertEqual(result['decision'],'REVIEW')

    def test_gateway_context_is_opt_in_authenticated_and_metadata_only(self):
        import asyncio
        from roguard.service import GatewayApp
        from roguard.observability import Monitor
        monitor=Monitor();app=GatewayApp(None,{'x'*32:{'tenant':'demo','role':'reader'}},context_model=Stub(),monitor=monitor)
        payload={'language':'ro','turns':[{'role':'minor','text':'PRIVATE_TEST_TEXT'}]}
        async def request(token):
            sent=[]
            async def receive():return {'type':'http.request','body':json.dumps(payload).encode()}
            async def send(message):sent.append(message)
            await app({'type':'http','method':'POST','path':'/v1/analyze-context','headers':[(b'authorization',('Bearer '+token).encode()),(b'content-type',b'application/json')]},receive,send)
            return sent
        denied=asyncio.run(request('wrong'));self.assertEqual(denied[0]['status'],401)
        accepted=asyncio.run(request('x'*32));self.assertEqual(accepted[0]['status'],200)
        self.assertNotIn('PRIVATE_TEST_TEXT',accepted[1]['body'].decode())
        self.assertNotIn('PRIVATE_TEST_TEXT',str(monitor.report('demo')))
        self.assertEqual(monitor.report('demo')['languages']['ro']['count'],1)
        with self.assertRaises(ValueError):GatewayApp(None,{'x'*32:{'tenant':'demo','role':'reader'}}).route('/v1/analyze-context','demo',payload)
        monitor.close()

    def test_model_rejects_corrupt_dimensions(self):
        import gzip,tempfile
        model=ContextModel('character');data=copy.deepcopy(model.spec);data['heads']['frame']['weights'][0].pop()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.json.gz';path.write_bytes(gzip.compress(json.dumps(data).encode()))
            with self.assertRaises(ValueError):ContextModel('character',artifact=path)


class ContextArtifactTests(unittest.TestCase):
    def test_packaged_artifacts_match_the_frozen_release(self):
        import hashlib
        from importlib import resources
        manifest=json.loads(Path('research/v7-1-context/release-freeze-7.1.1.json').read_text())
        for backend in ('character','semantic','joint'):
            data=resources.read_binary('roguard','context-'+backend+'.json.gz')
            self.assertEqual(hashlib.sha256(data).hexdigest(),manifest[backend+'.json.gz'])
        for name in ('context.py','context_signals.py','context.js'):
            self.assertEqual(hashlib.sha256(resources.read_binary('roguard',name)).hexdigest(),manifest[name])

class ContextReviewReasonTests(unittest.TestCase):
    def test_abstention_is_distinct_from_detected_concern(self):
        class Proposal(Stub):
            def predict(self,texts,language):
                return [{'supported':True,'heads':{'frame':{'label':'SELF_CURRENT','prediction_set':['SELF_CURRENT']},'signal':{'label':label,'prediction_set':[] if label is None else [label]}}} for label in (0,None,1)]
        result=analyze_context({'language':'uk','turns':[{'role':'minor','text':'Приклад'}]*3},Proposal())
        self.assertEqual(result['review_turn_indices'],[1,2])
        self.assertEqual(result['unresolved_turn_indices'],[1])
        self.assertEqual([r['review_kind'] for r in result['evidence']],['NO_REVIEW_SIGNAL','MODEL_UNCERTAINTY','CONCERN_SIGNAL'])
        self.assertEqual(result['evidence'][0]['review_question'],'')
        self.assertIn('without assuming harm',result['evidence'][1]['review_question'])

class ContextConditionalCalibrationTests(unittest.TestCase):
    def test_class_conditional_set_cannot_override_probability_winner(self):
        import math
        model=ContextModel('character')
        head=model.spec['heads']['signal']
        head['weights']=[[0.0]*len(head['weights'][0])]
        head['quantiles']={'ro':[0.95,0.05],'uk':[0.95,0.05]}
        for probability,label in ((0.1,0),(0.9,None),(0.99,1)):
            head['intercept']=[math.log(probability/(1-probability))]
            result=model.predict(['FOCUS minor: Exemplu fictiv.'],'ro')[0]['heads']['signal']
            self.assertEqual(result['label'],label)
            if probability==0.9:
                self.assertEqual(result['prediction_set'],[0])
                self.assertEqual(result['argmax'],1)

class ContextHeadDisagreementTests(unittest.TestCase):
    def test_positive_mention_with_ordinary_frame_requires_review(self):
        class Disagree(Stub):
            def predict(self,texts,language):
                return [{'supported':True,'heads':{'frame':{'label':'ORDINARY','prediction_set':['ORDINARY']},'signal':{'label':v,'prediction_set':[v]}}} for v in (0,1)]
        report=analyze_context({'language':'ro','turns':[{'role':'minor','text':'Exemplu fictiv.'}]*2},Disagree())
        self.assertEqual(report['review_turn_indices'],[1])
        self.assertEqual(report['unresolved_turn_indices'],[1])
        self.assertEqual(report['evidence'][1]['review_kind'],'MODEL_UNCERTAINTY')
        self.assertIn('disagree',report['evidence'][1]['review_question'])
        self.assertEqual(report['evidence'][0]['review_question'],'')
