import unittest
from roguard.context_signals import explicit_signals
from roguard.context import analyze_context


class ExplicitSignalTests(unittest.TestCase):
    def test_compositional_basic_needs_conditions(self):
        for language,parts in (
            ('ro',('Nu am mâncat','de două zile','și sunt singură acasă.')),
            ('uk',('Я не їла','вже два дні','і немає жодного дорослого поруч.')),
        ):
            self.assertIn('PROLONGED_UNMET_FOOD_AND_CARE',explicit_signals(' '.join(parts),language,'SELF_CURRENT'))
            for drop in range(3):
                self.assertEqual(explicit_signals(' '.join(p for i,p in enumerate(parts) if i!=drop),language,'SELF_CURRENT'),[])
            self.assertEqual(explicit_signals(' '.join(parts),language,'QUOTED'),[])

    def test_support_request_is_scoped_and_task_help_does_not_match(self):
        for lang,text in (('ro','Am nevoie de sprijin.'),('uk','Мені потрібна чиясь підтримка.')):
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT'),['EXPLICIT_SUPPORT_REQUEST'])
            self.assertEqual(explicit_signals(text,lang,'QUOTED'),[])
        for lang,text in (('ro','Am nevoie de sprijin la matematică.'),('uk','Мені потрібна підтримка з математикою.'),('ro','Mi-e foame, dar tata pregătește cina.'),('uk','Я сам у кімнаті та читаю книжку.')):
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT'),[])

    def test_rule_can_request_review_without_rewriting_model_output(self):
        class Negative:
            backend='test';identity='test'
            def predict(self,texts,language):
                return [{'supported':True,'heads':{'frame':{'label':'SELF_CURRENT','prediction_set':['SELF_CURRENT']},'signal':{'label':0,'prediction_set':[0]}}} for text in texts]
        report=analyze_context({'language':'uk','turns':[{'role':'minor','text':'Мені потрібна підтримка.'}]},Negative())
        self.assertEqual(report['decision'],'REVIEW')
        self.assertEqual(report['evidence'][0]['mention'],0)
        self.assertEqual(report['evidence'][0]['rule_signals'],['EXPLICIT_SUPPORT_REQUEST'])
        self.assertFalse(report['external_action_taken'])
        self.assertFalse(report['authority_from_model'])

class ContextTransitionSignalTests(unittest.TestCase):
    def test_quotation_conflict_requires_all_three_conditions(self):
        for lang,parts in (('ro',('Încheiem scena.','Vorbesc despre mine, în realitate.')),('uk',('Припинимо рольову гру.','Особисто я хочу розповісти про себе.'))):
            text=' '.join(parts)
            self.assertEqual(explicit_signals(text,lang,'QUOTED',1),['PERSONAL_CONTEXT_CONTRADICTS_QUOTATION'])
            self.assertEqual(explicit_signals(text,lang,'QUOTED',0),[])
            for part in parts:self.assertEqual(explicit_signals(part,lang,'QUOTED',1),[])

    def test_emotional_qualification_without_matching_homework(self):
        self.assertEqual(explicit_signals('Am nevoie de sprijin pentru sentimentele mele.', 'ro','SELF_CURRENT'),['EXPLICIT_SUPPORT_REQUEST'])
        self.assertEqual(explicit_signals('Мені потрібна підтримка через мої переживання.', 'uk','SELF_CURRENT'),['EXPLICIT_SUPPORT_REQUEST'])
        self.assertEqual(explicit_signals('Am nevoie de sprijin pentru tema mea.', 'ro','SELF_CURRENT'),[])
