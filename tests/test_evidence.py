"""Evidence invariants and permission boundaries; no model accuracy assertions."""
from copy import deepcopy
import json
import unittest
from roguard.evidence import ONTOLOGY, EvidenceCascade, EvidenceState, analyze_evidence, disposition, plan_response, units
from roguard.flow import FlowLedger


def prediction(**changes):
    labels={'subject':'SELF','time':'CURRENT','assertion':'AFFIRMED','source':'DIRECT','relation':'NEW','concern':'1'}
    labels.update(changes)
    return {'supported':True,'reason':'MODEL_PROPOSAL','attributes':{k:{'label':v,'argmax':v,'prediction_set':[v] if v is not None else list(ONTOLOGY[k])} for k,v in labels.items()}}

class Stub:
    identity='stub'
    def __init__(self,p=None):self.p=p or prediction();self.calls=0
    def predict(self,texts,language):self.calls+=len(texts);return [deepcopy(self.p) for _ in texts]

class EvidenceTests(unittest.TestCase):
    def test_localized_summary_survives_legacy_stdout_encoding(self):
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        for language,text in [('ro','Mă simt singur.'),('uk','Мені самотньо.')]:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'input.json'
                path.write_text(json.dumps({'language':language,'turns':[{'role':'minor','text':text}]},ensure_ascii=False),encoding='utf-8')
                result=subprocess.run([sys.executable,'-m','roguard.evidence',str(path)],
                    env=dict(os.environ,PYTHONIOENCODING='cp1252'),capture_output=True,check=True)
                report=json.loads(result.stdout.decode('ascii'))
                self.assertEqual(report['language'],language)
                self.assertTrue(report['evidence'][0]['interpretation_summary']['label'])

    def test_explanation_distinguishes_uncertainty_and_no_signal(self):
        for language in ('ro', 'uk'):
            payload={'language':language,'turns':[{'role':'minor','text':'example'}]}
            for p,status in [(prediction(concern=None),'clarification_needed'),
                             (prediction(concern='0'),'no_signal_detected'),
                             (prediction(),'review_requested')]:
                report=analyze_evidence(payload,Stub(p))
                summary=report['evidence'][0]['interpretation_summary']
                self.assertEqual(summary['status'],status)
                self.assertFalse(summary['permits_notification'])
                self.assertFalse(summary['verified'])
                self.assertEqual('concern' in summary['unresolved_attributes'],status=='clarification_needed')

    def test_original_offsets_and_no_plaintext(self):
        text='  Ma\u0306 simt singur.  \n   Altă propoziție. '
        p={'language':'ro','turns':[{'role':'minor','text':text}]}
        spans=units(p)
        self.assertEqual(len(spans),2)
        for s in spans:self.assertEqual(text[s['start']:s['end']],s['text'])
        report=analyze_evidence(p,Stub())
        self.assertNotIn('simt',json.dumps(report));self.assertFalse(report['external_action_taken'])
        self.assertTrue(all(not e['verified'] for e in report['evidence']))

    def test_canonically_equivalent_bilingual_text_keeps_same_predictions(self):
        import unicodedata
        from roguard.evidence import EvidenceModel
        model=EvidenceModel()
        for language,text in [('ro','Îmi este teamă și aș vrea să mă asculte cineva.'),
                              ('uk','Мій друг почувається самотнім і хоче з кимось поговорити.')]:
            decomposed=unicodedata.normalize('NFD',text)
            self.assertNotEqual(text,decomposed)
            self.assertEqual(model.predict([text],language),model.predict([decomposed],language))
            report=analyze_evidence({'language':language,'turns':[{'role':'minor','text':decomposed}]},model)
            ref=report['evidence'][0]['reference']
            self.assertEqual(decomposed[ref['start']:ref['end']],decomposed)

    def test_bounds(self):
        with self.assertRaises(ValueError):units({'language':'ro','turns':[{'role':'minor','text':'a\n'*49}]})

    def test_cascade_abstains_on_decided_conflict(self):
        primary=Stub(prediction(time=None));other=Stub(prediction(subject='OTHER',source='REPORTED'))
        result=EvidenceCascade(primary,other).predict(['example'],'ro')[0]
        self.assertIsNone(result['attributes']['subject']['label'])
        self.assertEqual(result['attributes']['time']['label'],'CURRENT')
        self.assertIn('subject',result['disagreements'])
        self.assertTrue(result['challenger_used'])

    def test_challenger_cannot_hide_primary_set_argmax_conflict(self):
        p=prediction(subject=None)
        p['attributes']['subject'].update(prediction_set=['FICTIONAL'],argmax='SELF')
        q=prediction(subject='FICTIONAL',source='FICTION')
        merged=EvidenceCascade(Stub(p),Stub(q)).predict(['text'],'ro')[0]
        self.assertIsNone(merged['attributes']['subject']['label'])

    def test_cascade_does_not_call_challenger_for_resolved_units(self):
        other=Stub();EvidenceCascade(Stub(),other).predict(['example'],'ro')
        self.assertEqual(other.calls,0)

    def test_no_support_claim_from_uncertainty(self):
        self.assertEqual(disposition(prediction(concern=None)['attributes'])[0],'REVIEW_AND_CONTINUE')
        self.assertEqual(disposition(prediction(assertion='NEGATED')['attributes'])[0],'CONTINUE')
        self.assertEqual(disposition(prediction(assertion='RETRACTED')['attributes'])[0],'REVIEW_AND_CONTINUE')

    def test_host_revision_keeps_history(self):
        p={'language':'ro','turns':[{'role':'minor','text':'Prima afirmație.'},{'role':'minor','text':'O corectez.'}]}
        r=analyze_evidence(p,Stub());state=EvidenceState();state.update(r)
        self.assertEqual(state.snapshot()['active_review_ids'],['t0_s0','t1_s0'])
        revised=state.confirm_revision('t1_s0','t0_s0',reviewer_id='host_reviewer')
        self.assertEqual(revised['active_review_ids'],['t1_s0']);self.assertEqual(len(revised['report']['evidence']),2)
        with self.assertRaises(ValueError):state.confirm_revision('t0_s0','t1_s0',reviewer_id='host_reviewer')
        changed=deepcopy(r);changed['evidence'][0]['attributes']['concern']['label']='0'
        with self.assertRaises(ValueError):state.update(changed)
        state.close();self.assertIsNone(state.snapshot()['report'])
        with self.assertRaises(ValueError):state.update(r)

    def test_review_index_cannot_silently_drop_historical_review(self):
        report=analyze_evidence({'language':'ro','turns':[{'role':'minor','text':'Prima.'}]},Stub())
        state=EvidenceState();state.update(report)
        changed=deepcopy(report);changed['review_ids']=[]
        with self.assertRaises(ValueError):state.update(changed)
        self.assertEqual(state.snapshot()['active_review_ids'],['t0_s0'])

    def test_append_does_not_clear_prior_reviews(self):
        state=EvidenceState();first=analyze_evidence({'language':'ro','turns':[{'role':'minor','text':'Prima.'}]},Stub())
        state.update(first)
        second=deepcopy(first);second['evidence'].append(dict(deepcopy(first['evidence'][0]),id='t1_s0',reference={'turn_index':1}));second['review_ids'].append('t1_s0')
        self.assertEqual(len(state.update(second)['active_review_ids']),2)

    def test_permissions_and_requirements_are_both_enforced(self):
        report=analyze_evidence({'language':'uk','turns':[{'role':'minor','text':'Мені самотньо.'}]},Stub())
        ledger=FlowLedger();ledger.add_source('private','sensitive source');ledger.add_source('public','task material')
        ledger.permission('public','tutor','minor','use','grant')
        candidates=[{'id':'leak','artifacts':['private'],'covers':['task','support','review'],'disruption':0},
                    {'id':'dismiss','artifacts':['public'],'covers':['task'],'disruption':0},
                    {'id':'useful','artifacts':['public'],'covers':['task','support','review'],'disruption':2}]
        result=plan_response(report,candidates,ledger,purpose='tutor',recipient='minor')
        self.assertEqual(result['selected'],'useful');self.assertFalse(result['execution_authorized'])
        ledger.permission('public','tutor','minor','use','revoke')
        self.assertEqual(plan_response(report,candidates,ledger,purpose='tutor',recipient='minor')['status'],'NO_PERMITTED_RESPONSE')

    def test_empty_minor_input_has_no_review_evidence(self):
        report=analyze_evidence({'language':'ro','turns':[{'role':'assistant','text':'Salut.'}]},Stub())
        self.assertEqual(report['evidence'],[])


class GraphTests(unittest.TestCase):
    def test_repeated_claim_negation_creates_unverified_edge(self):
        class Sequence(Stub):
            def predict(self,texts,language):return [prediction(),prediction(assertion='NEGATED')]
        p={'language':'ro','turns':[{'role':'minor','text':'Confirm: «Mă simt singur».'},{'role':'minor','text':'Neg: «Mă simt singur».'}]}
        report=analyze_evidence(p,Sequence())
        self.assertEqual(report['candidate_edges'][0]['relation'],'POSSIBLE_CONTRADICTION')
        self.assertFalse(report['candidate_edges'][0]['verified'])
        for e in report['evidence']:
            r=e['reference'];self.assertEqual(p['turns'][r['turn_index']]['text'][r['candidate_claim_start']:r['candidate_claim_end']],'Mă simt singur')
        self.assertIn('t0_s0',report['review_ids'])

    def test_canonical_claim_equivalence_preserves_original_offsets(self):
        import unicodedata
        claim='Îmi este teamă.'
        other=unicodedata.normalize('NFD',claim)
        payload={'language':'ro','turns':[{'role':'minor','text':'«'+claim+'»'},
                                        {'role':'minor','text':'«'+other+'»'}]}
        report=analyze_evidence(payload,Stub())
        self.assertEqual(report['candidate_edges'][0]['relation'],'REPEATED_ASSERTION')
        self.assertFalse(report['candidate_edges'][0]['verified'])
        for unit,original in zip(report['evidence'],(claim,other)):
            ref=unit['reference']
            self.assertEqual(payload['turns'][ref['turn_index']]['text'][ref['candidate_claim_start']:ref['candidate_claim_end']],original)

    def test_model_disagreement_cannot_create_verified_fact(self):
        a=prediction(subject=None)
        class Sequence(Stub):
            def predict(self,texts,language):return [a,a]
        p={'language':'ro','turns':[{'role':'minor','text':'«Același text»'},{'role':'minor','text':'«Același text»'}]}
        self.assertEqual(analyze_evidence(p,Sequence())['candidate_edges'],[])

class EvidenceGatewayTests(unittest.TestCase):
    def test_endpoint_requires_auth_and_explicit_configuration(self):
        import asyncio
        from roguard.service import GatewayApp
        payload={'language':'ro','turns':[{'role':'minor','text':'PRIVATE_SENTINEL'}]}
        app=GatewayApp(None,{'x'*32:{'tenant':'demo','role':'reader'}},evidence_model=Stub())
        async def request(token):
            sent=[]
            async def receive():return {'type':'http.request','body':json.dumps(payload).encode()}
            async def send(m):sent.append(m)
            await app({'type':'http','method':'POST','path':'/v1/analyze-evidence','headers':[(b'authorization',('Bearer '+token).encode()),(b'content-type',b'application/json')]},receive,send)
            return sent
        self.assertEqual(asyncio.run(request('bad'))[0]['status'],401)
        accepted=asyncio.run(request('x'*32));self.assertEqual(accepted[0]['status'],200)
        self.assertNotIn('PRIVATE_SENTINEL',accepted[1]['body'].decode())
        with self.assertRaises(ValueError):GatewayApp(None,{'x'*32:{'tenant':'demo','role':'reader'}}).route('/v1/analyze-evidence','demo',payload)

    def test_packaged_character_model_and_unsupported_script(self):
        from roguard.evidence import EvidenceModel
        model=EvidenceModel()
        p=analyze_evidence({'language':'uk','turns':[{'role':'minor','text':'Meni samotno'}]},model)
        self.assertFalse(p['evidence'][0]['supported']);self.assertTrue(p['review_ids'])
        self.assertEqual(p['evidence'][0]['model_reason'],'UKRAINIAN_SCRIPT_REQUIRED')

    def test_report_has_no_mutable_alias_with_state(self):
        r=analyze_evidence({'language':'ro','turns':[{'role':'minor','text':'Afirm ceva.'}]},Stub())
        state=EvidenceState();snapshot=state.update(r);snapshot['report']['review_ids'].clear();r['review_ids'].clear()
        self.assertEqual(state.snapshot()['active_review_ids'],['t0_s0'])

class GuardedEvidenceTests(unittest.TestCase):
    def test_released_review_is_retained_without_rewriting_attributes(self):
        from roguard.evidence import GuardedEvidence
        class Context:
            identity='released';backend='joint'
            def predict(self,texts,language):
                self.inputs=texts
                return [{'supported':True,'heads':{'frame':{'label':'SELF_CURRENT'},'signal':{'label':1}}} for t in texts]
        context=Context();primary=Stub(prediction(concern='0'))
        payload={'language':'ro','turns':[{'role':'minor','text':'Prima propoziție.\nA doua propoziție.'}]}
        report=analyze_evidence(payload,GuardedEvidence(primary,context))
        self.assertEqual(len(context.inputs),1)
        self.assertIn('A doua propoziție.',context.inputs[0])
        for e in report['evidence']:
            self.assertEqual(e['attributes']['concern']['label'],'0')
            self.assertEqual(e['attribute_action'],'CONTINUE')
            self.assertEqual(e['proposed_action'],'REVIEW_AND_CONTINUE')
            self.assertTrue(e['review_floor']['review_recommended'])
        self.assertFalse(report['external_action_taken'])

    def test_no_released_signal_cannot_cancel_new_support_signal(self):
        from roguard.evidence import GuardedEvidence
        class Context:
            identity='released';backend='joint'
            def predict(self,texts,language):return [{'supported':True,'heads':{'frame':{'label':'ORDINARY'},'signal':{'label':0}}} for t in texts]
        report=analyze_evidence({'language':'ro','turns':[{'role':'minor','text':'Afirmație.'}]},GuardedEvidence(Stub(),Context()))
        self.assertEqual(report['evidence'][0]['proposed_action'],'SUPPORT_AND_REVIEW')


class EnvelopeTests(unittest.TestCase):
    def test_set_argmax_conflict_cannot_silently_resolve_to_fiction(self):
        from roguard.evidence import decision_envelope
        attrs=prediction(subject=None,source=None)['attributes']
        attrs['subject'].update(prediction_set=['FICTIONAL'],argmax='SELF')
        attrs['source'].update(prediction_set=['FICTION'],argmax='DIRECT')
        env=decision_envelope(attrs)
        self.assertIn('SUPPORT_AND_REVIEW',env['possible_actions'])
        self.assertFalse(env['stable'])
        self.assertEqual(disposition(attrs)[0],'REVIEW_AND_CONTINUE')

    def test_uncertain_time_does_not_force_extra_review_when_action_is_identical(self):
        from roguard.evidence import decision_envelope
        attrs=prediction(time=None)['attributes'];attrs['time']['prediction_set']=['CURRENT','PAST']
        env=decision_envelope(attrs)
        self.assertTrue(env['stable']);self.assertEqual(env['possible_actions'],['SUPPORT_AND_REVIEW'])
        self.assertEqual(env['hypothesis_count'],2)

    def test_negation_or_affirmation_exposes_both_witnesses(self):
        from roguard.evidence import decision_envelope
        attrs=prediction(assertion=None)['attributes'];attrs['assertion']['prediction_set']=['AFFIRMED','NEGATED']
        env=decision_envelope(attrs)
        self.assertEqual(set(env['possible_actions']),{'CONTINUE','SUPPORT_AND_REVIEW'})
        self.assertEqual(env['reviewer_questions'][0]['attribute'],'assertion')
        self.assertFalse(env['real_world_certainty'])

    def test_fiction_does_not_need_temporal_certainty_but_cannot_clear_retraction(self):
        p=prediction(subject='FICTIONAL',source='FICTION',time=None)
        self.assertEqual(disposition(p['attributes'])[0],'CONTINUE')
        p['attributes']['relation']={'label':'RETRACTS'}
        self.assertEqual(disposition(p['attributes'])[0],'REVIEW_AND_CONTINUE')


class FrozenEvidenceArtifacts(unittest.TestCase):
    def test_bundled_models_match_frozen_candidate(self):
        import hashlib
        from pathlib import Path
        freeze=json.loads(Path('research/v8-evidence/bundled-models.json').read_text())
        for name in ('character','semantic'):
            path=Path('src/roguard/evidence-'+name+'.json.gz')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),freeze['files'][str(path)])

class HypotheticalTests(unittest.TestCase):
    def test_hypothetical_concern_is_not_automatically_dismissed(self):
        self.assertEqual(disposition(prediction(time='HYPOTHETICAL')['attributes'])[0],'REVIEW_AND_CONTINUE')
        self.assertEqual(disposition(prediction(time='HYPOTHETICAL',subject='FICTIONAL',source='FICTION')['attributes'])[0],'CONTINUE')



class EvidenceSessionTests(unittest.TestCase):
    def test_append_encodes_each_immutable_prefix_once(self):
        from roguard.evidence import EvidenceSession
        model=Stub()
        with EvidenceSession('ro',model) as session:
            for i in range(12):session.append('minor','Mesaj '+str(i))
            self.assertEqual(model.calls,12)
            self.assertEqual(len(session.model.cache),12)
            self.assertEqual(len(session.state.snapshot()['report']['evidence']),12)
        self.assertEqual(session.model.cache,{})

    def test_subspan_derivation_cannot_declassify_whole_turn(self):
        from roguard.evidence import EvidenceSession
        with EvidenceSession('ro',Stub()) as session:
            session.append('minor','O afirmație.\nAltă afirmație.')
            snapshot=session.state.snapshot();eid=snapshot['report']['evidence'][1]['id']
            derived=session.derive_from_evidence('response','Host-reviewed response text.',[eid])
            self.assertEqual(derived['sources'],['turn_0'])
            c=[{'id':'response','artifacts':['response'],'covers':['task','support','review'],'disruption':1}]
            self.assertEqual(session.propose_response(c,purpose='tutor',recipient='minor')['status'],'NO_PERMITTED_RESPONSE')
            session.ledger.permission('turn_0','tutor','minor','use','grant')
            self.assertEqual(session.propose_response(c,purpose='tutor',recipient='minor')['selected'],'response')
            session.ledger.permission('turn_0','tutor','minor','use','revoke')
            self.assertEqual(session.propose_response(c,purpose='tutor',recipient='minor')['status'],'NO_PERMITTED_RESPONSE')
        with self.assertRaises(ValueError):session.append('minor','After close.')

    def test_review_revision_changes_plan_but_not_permissions_or_history(self):
        from roguard.evidence import EvidenceSession
        class ByText(Stub):
            def predict(self,texts,language):return [prediction(concern='0' if 'clarification' in t['text'] else '1') for t in texts]
        with EvidenceSession('ro',ByText()) as session:
            session.append('minor','First claim.');session.append('minor','Host-reviewed clarification.')
            session.derive_from_evidence('response','Response.', ['t0_s0','t1_s0'])
            for source in ('turn_0','turn_1'):session.ledger.permission(source,'tutor','minor','use','grant')
            c=[{'id':'response','artifacts':['response'],'covers':['task'],'disruption':1}]
            self.assertEqual(session.propose_response(c,purpose='tutor',recipient='minor')['status'],'NO_PERMITTED_RESPONSE')
            revision=session.ledger.revision
            session.state.confirm_revision('t1_s0','t0_s0',reviewer_id='reviewer')
            self.assertEqual(session.ledger.revision,revision)
            self.assertEqual(len(session.state.snapshot()['report']['evidence']),2)
            self.assertEqual(session.propose_response(c,purpose='tutor',recipient='minor')['selected'],'response')

if __name__=='__main__':unittest.main()
