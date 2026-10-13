"""Conversation boundaries, independent authority, uncertainty and text privacy."""
import copy
import json
import math
import unittest
from roguard.conversation import analyze_conversation, analyze_batch, compare_reports, ConversationScorer, validate
from roguard.response_review import review_response
from roguard.message_model import SyntheticMessageModel

class FixedScorer:
    backend='fast'
    def __init__(self,value):self.value=value
    def scores(self,texts,language):return [self.value]*len(texts)
    def thresholds(self,language):return {'cutoff':.5,'abstain_margin':.08}


def sample(language='ro'):
    return {'case_id':'case_1','language':language,'turns':[{'role':'minor','text':
        'Vorbesc despre mine și am nevoie de sprijin.' if language=='ro' else 'Говорю про себе і потребую підтримки.'}]}

class ConversationTests(unittest.TestCase):
    def test_uncertainty_boundaries_and_nan_do_not_claim_negative(self):
        for score,expected in [(.1,'NO_MODEL_SIGNAL'),(.5,'INSUFFICIENT_EVIDENCE'),(.9,'REVIEW_SUPPORT_SIGNAL'),(math.nan,'INSUFFICIENT_EVIDENCE')]:
            self.assertEqual(analyze_conversation(sample(),FixedScorer(score))['signal']['decision'],expected)

    def test_text_not_echoed_and_no_action(self):
        p=sample();p['draft']='Putem discuta în ritmul tău.'
        r=analyze_conversation(p,FixedScorer(.9))
        self.assertNotIn(p['turns'][0]['text'],json.dumps(r,ensure_ascii=False))
        self.assertNotIn(p['draft'],json.dumps(r,ensure_ascii=False))
        self.assertFalse(r['external_action_taken']);self.assertFalse(r['authority_from_model'])
        ref=r['signal']['evidence_references'][0]
        self.assertEqual(ref['end'],len(p['turns'][0]['text']))

    def test_model_cannot_grant_revoked_permission_or_change_audit(self):
        p=sample();p['contract']={'language':'ro','permission':{'principal':'minor','item':'x','purpose':'a','recipient':'guardian','at':2,'proposed_use':True,'use_required':True,'events':[{'sequence':1,'principal':'minor','item':'x','purpose':'a','recipient':'guardian','action':'revoke'}]}}
        p['audit']={'language':'ro','routing':{'principal':'minor','item':'x','purpose':'a','proposed_recipient':'guardian','recipient_roles':['minor','guardian'],'allowed_by_scope':[{'principal':'minor','item':'x','purpose':'a','recipients':['guardian']}],'recipient_may_be_source_of_concern':'unknown'}}
        low=analyze_conversation(p,FixedScorer(.1));high=analyze_conversation(p,FixedScorer(.9))
        self.assertEqual(low['declared_contract'],high['declared_contract'])
        self.assertEqual(low['uncertainty_audit'],high['uncertainty_audit'])
        self.assertEqual(low['action_review'],high['action_review'])
        self.assertEqual(high['action_review']['status'],'HUMAN_REVIEW_REQUIRED')
        self.assertNotEqual(low['declared_contract']['findings'][0]['status'],'pass')
        self.assertTrue(high['uncertainty_audit']['categories'][0]['next_questions'])

    def test_unsupported_language_script_and_roles_abstain(self):
        p=sample();p['language']='en'
        self.assertEqual(analyze_conversation(p)['signal']['decision'],'INSUFFICIENT_EVIDENCE')
        p=sample('uk');p['turns'][0]['text']='This is an English language sentence.'
        self.assertIn('SCRIPT_MISMATCH_OR_MIXTURE',analyze_conversation(p)['signal']['reason_codes'])
        for role in ['adult','unknown','other']:
            p=sample();p['turns'][0]['role']=role
            self.assertEqual(analyze_conversation(p)['signal']['decision'],'INSUFFICIENT_EVIDENCE')

    def test_role_order_and_context_are_kept(self):
        class Capture(FixedScorer):
            def scores(self,texts,language):
                self.seen=getattr(self,'seen',[])+texts
                return super().scores(texts,language)
        p=sample();p['turns'].append({'role':'assistant','text':'Putem continua.'});p['turns'].append({'role':'minor','text':'Mai bine mă opresc.'})
        scorer=Capture(.9);analyze_conversation(p,scorer)
        self.assertIn('assistant: Putem continua.\nminor: Mai bine mă opresc.',scorer.seen[0])

    def test_bounded_and_strict_input(self):
        for edit in [lambda p:p.update(extra=True),lambda p:p.update(turns=[]),lambda p:p.update(turns=p['turns']*17),lambda p:p.update(case_id='private words here'),lambda p:p.update(revision='without draft'),lambda p:p.update(contract={'language':'uk'})]:
            p=sample();edit(p)
            with self.assertRaises(ValueError):validate(p)
        p=sample();p['turns'][0]['text']='x'*2001
        with self.assertRaises(ValueError):validate(p)

    def test_batch_no_partial_return_and_comparison_checks_identity(self):
        with self.assertRaises(ValueError):analyze_batch([sample(),{'language':'ro'}],FixedScorer(.9))
        a=analyze_conversation(sample(),FixedScorer(.1));b=analyze_conversation(sample(),FixedScorer(.9))
        self.assertEqual(len(compare_reports([a],[b])['decision_changes']),1)
        self.assertFalse(compare_reports([a],[b])['accuracy_change_established'])
        b['case_id']='different'
        with self.assertRaises(ValueError):compare_reports([a],[b])
        with self.assertRaises(ValueError):compare_reports([a,a],[a,a])

    def test_real_packaged_fast_models_and_cached_v3(self):
        for language in ['ro','uk']:
            r=analyze_conversation(sample(language))
            self.assertIsNotNone(r['signal']['synthetic_score'])
            self.assertEqual(len(r['signal']['model']['fast_sha256']),64)
            self.assertIs(SyntheticMessageModel.packaged(language),SyntheticMessageModel.packaged(language))

    def test_optional_semantic_rejects_missing_base_files(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as path:
            with self.assertRaises((OSError,ImportError,ValueError)):ConversationScorer('semantic',path)

class ResponseTests(unittest.TestCase):
    def test_review_patterns_in_both_languages(self):
        for language,text in [('ro','Totul va fi bine. Spune-mi toate detaliile.'),('uk','Усе буде добре. Розкажи всі подробиці.')]:
            r=review_response(text,language)
            self.assertEqual({f['code'] for f in r['findings']},{'UNSUPPORTED_GUARANTEE','DETAIL_PRESSURE'})
            for f in r['findings']:self.assertTrue(text[f['start']:f['end']])
            self.assertFalse(r['safe_response_established'])

    def test_absence_is_not_certification_and_complexity_is_descriptive(self):
        r=review_response('Putem vorbi în ritmul tău.','ro')
        self.assertEqual(r['status'],'NO_LISTED_PATTERN');self.assertFalse(r['safe_response_established'])
        self.assertFalse(r['complexity']['reading_age_estimated'])
        self.assertGreater(r['complexity']['words'],0)

    def test_revision_warns_about_negation_and_numbers(self):
        r=review_response('Nu trimite 2 mesaje.','ro','Trimite 3 mesaje.')
        self.assertTrue(r['revision']['number_tokens_changed']);self.assertTrue(r['revision']['negation_count_changed'])
        self.assertFalse(r['revision']['meaning_preservation_established'])

if __name__=='__main__':unittest.main()
