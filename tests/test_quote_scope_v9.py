import unittest
from training.quote_scope_v9 import abstract_quotes


class QuoteScopeTests(unittest.TestCase):
    def test_preserves_denial_double_negation_and_unquoted_content(self):
        for text in ('Neg afirmația: «un mesaj». Dar cer sprijin.',
                     'Nu neg afirmația: «un mesaj».',
                     'Не заперечую твердження: «повідомлення».'):
            result=abstract_quotes(text)
            self.assertIn('QUOTED_CONTENT',result)
            self.assertEqual(result.split(' QUOTED_CONTENT ')[0],text.split('«')[0])
        self.assertTrue(abstract_quotes('Neg afirmația: «text». Dar cer sprijin.').endswith('Dar cer sprijin.'))

    def test_unclosed_quotes_remain_and_closed_styles_are_supported(self):
        for text in ('Un text «neterminat','Un text "neterminat','Un text „neterminat'):
            self.assertEqual(abstract_quotes(text),text)
        for text in ('«text»','„text”','„text“','“text”','"text"'):
            self.assertEqual(abstract_quotes(text),' QUOTED_CONTENT ')
