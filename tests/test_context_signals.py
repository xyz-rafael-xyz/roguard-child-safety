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

class NegatedSignalTests(unittest.TestCase):
    def test_negated_support_does_not_become_an_affirmative_rule_signal(self):
        for lang,text in (('ro','Nu am nevoie de sprijin.'),('ro','N-am nevoie de sprijin.'),('ro','Nu   am nevoie de sprijin.'),('ro','Nu cred că am nevoie de sprijin.'),('uk','Мені не потрібна підтримка.'),('uk','Не думаю, що мені потрібна підтримка.')):
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT',0),[],text)
        for lang,text in (('ro','Nu mi-e bine. Am nevoie de sprijin.'),('ro','Nu mereu, dar am nevoie de sprijin.'),('uk','Не все гаразд. Мені потрібна підтримка.')):
            self.assertIn('EXPLICIT_SUPPORT_REQUEST',explicit_signals(text,lang,'SELF_CURRENT',0))

    def test_negated_solitude_and_unrelated_nobody_are_not_absent_care(self):
        for lang,text in (('ro','Nu am mâncat de ieri, dar nu sunt singur.'),('ro','Nu am mâncat de ieri, dar nu mai sunt singură.'),('ro','Nu am mâncat de ieri; nu este nimeni în pericol.'),('uk','Я не їв від учора, але я не сам.'),('uk','Я не їв від учора; немає нікого злого поруч.')):
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT',0),[],text)
        for lang,text in (('ro','Nu am mâncat de ieri și sunt singur.'),('uk','Я не їв від учора і я сам.'),('ro','Nu am mâncat de ieri și nu este niciun adult acasă.')):
            self.assertIn('PROLONGED_UNMET_FOOD_AND_CARE',explicit_signals(text,lang,'SELF_CURRENT',0))

    def test_negated_exit_from_fiction_cannot_override_quotation(self):
        self.assertEqual(explicit_signals('Nu oprim ficțiunea. Personajul vorbește despre mine.', 'ro','QUOTED',1),[])
        self.assertEqual(explicit_signals('Не зупинимо вигадану історію. Особисто я читаю репліку.', 'uk','QUOTED',1),[])


class DeniedClaimSignalTests(unittest.TestCase):
    def test_denial_of_food_claim_is_not_asserted_deprivation(self):
        cases=[('ro','Nu este adevărat că nu am mâncat de trei zile. Sunt singur acasă.'),
               ('ro','Este fals că nu am mâncat de ieri. Sunt singură acasă.'),
               ('uk','Неправда, що я не їв кілька днів. Я сам удома.'),
               ('uk','Заперечую, що я не їла від учора. Я сама вдома.')]
        for lang,text in cases:
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT',0),[],text)

    def test_explicit_denial_scopes_support_and_care_claims(self):
        cases=[('ro','Este fals că am nevoie de sprijin.'),
               ('ro','Resping ideea că am nevoie de sprijin.'),
               ('uk','Неправда, що мені потрібна підтримка.'),
               ('uk','Заперечую, що мені потрібна підтримка.'),
               ('ro','Nu am mâncat de ieri. Este fals că sunt singur.'),
               ('uk','Я не їв від учора. Неправда, що я сам.')]
        for lang,text in cases:
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT',0),[],text)

    def test_denial_does_not_leak_into_later_affirmed_clause(self):
        cases=[('ro','Este fals că am terminat tema. Am nevoie de sprijin.'),
               ('ro','Nu este adevărat, dar am nevoie de sprijin.'),
               ('uk','Це неправда. Мені потрібна підтримка.'),
               ('uk','Це неправда, але мені потрібна підтримка.')]
        for lang,text in cases:
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT',0),['EXPLICIT_SUPPORT_REQUEST'],text)
        for lang,text in [('ro','Nu am mâncat de ieri și sunt singur.'),
                          ('uk','Я не їв від учора і я сам.'),
                          ('ro','Nu știu ce să fac, nu am mâncat de ieri și sunt singur.'),
                          ('uk','Не знаю, що робити, я не їв від учора і я сам.')]:
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT',0),['PROLONGED_UNMET_FOOD_AND_CARE'])


class FoodRecipientScopeTests(unittest.TestCase):
    def test_lack_of_pet_food_is_not_personal_food_deprivation(self):
        for lang,text in [('ro','Nu am hrană pentru pești de ieri și sunt singur în cameră.'),
                          ('ro','Nu mai am mâncare pentru pisici de două zile și sunt singură.'),
                          ('uk','У мене немає їжі для рибок від учора і я сам у кімнаті.'),
                          ('uk','У мене закінчилася їжа для кота кілька днів тому і я сама.')]:
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT',0),[],text)

    def test_personal_deprivation_is_retained_even_when_pets_are_mentioned(self):
        for lang,text in [('ro','Nu am hrană pentru mine de ieri și sunt singur.'),
                          ('uk','У мене немає їжі для мене від учора і я сам.'),
                          ('ro','Nu am mâncat de ieri și sunt singur. Am hrană pentru pisici.'),
                          ('uk','Я не їв від учора і я сам. Маю їжу для кота.')]:
            self.assertEqual(explicit_signals(text,lang,'SELF_CURRENT',0),['PROLONGED_UNMET_FOOD_AND_CARE'],text)
