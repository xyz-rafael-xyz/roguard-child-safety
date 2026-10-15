"""Training hygiene: prevent equivalent inputs from crossing partition boundaries."""
import copy
import unicodedata
import unittest
from training.review_data_v8 import isolate_partitions


def row(name, text, concern='0'):
    return {'case_id': name, 'language': 'ro', 'turns': [{'role': 'minor', 'text': text}],
            'focus': 0, 'attributes': {'subject': 'SELF', 'time': 'CURRENT',
            'assertion': 'AFFIRMED', 'source': 'DIRECT', 'relation': 'NEW', 'concern': concern}}


class ReviewPartitionTests(unittest.TestCase):
    def test_normalized_duplicates_stay_in_first_partition(self):
        partitions={'train':[row('a','Învăț adunarea.'),row('b','Învăț adunarea.')],
                    'dev':[row('c',unicodedata.normalize('NFD','Învăț adunarea.')),row('d','O altă temă.')],
                    'cal':[row('e','Învăț  adunarea.'),row('f','O altă temă.'),row('g','Desenez.') ]}
        original=copy.deepcopy(partitions)
        cleaned,removed=isolate_partitions(partitions)
        self.assertEqual({s:[r['case_id'] for r in rs] for s,rs in cleaned.items()},
                         {'train':['a'],'dev':['d'],'cal':['g']})
        self.assertEqual(removed,{'train':['b'],'dev':['c'],'cal':['e','f']})
        self.assertEqual(partitions,original)

    def test_same_input_with_conflicting_target_is_an_error(self):
        with self.assertRaisesRegex(ValueError,'Conflicting review labels'):
            isolate_partitions({'train':[row('a','Un text.')],'dev':[row('b','Un text.','1')],'cal':[]})

    def test_different_prior_scope_is_a_different_model_input(self):
        a=row('a','Un text.');b=row('b','Un text.')
        b['turns'].insert(0,{'role':'minor','text':'Vorbesc despre un personaj.'});b['focus']=1
        cleaned,_=isolate_partitions({'train':[a],'dev':[b],'cal':[]})
        self.assertEqual(len(cleaned['dev']),1)


class HistoricalTrainingTests(unittest.TestCase):
    def test_warm_start_history_cannot_reenter_evaluation(self):
        history=[row('old','Învăț adunarea.')]
        partitions={'train':[row('new','Desenez.')],
                    'dev':[row('leak','Învăț adunarea.'),row('clean','O altă temă.')],
                    'cal':[row('leak-cal','Învăț adunarea.')]}
        cleaned,removed=isolate_partitions(partitions,history)
        self.assertEqual([r['case_id'] for r in cleaned['dev']],['clean'])
        self.assertEqual(removed['cal'],['leak-cal'])

    def test_rehearsal_of_prior_training_is_allowed_but_relabeling_is_not(self):
        history=[row('old','Un text.')]
        cleaned,_=isolate_partitions({'train':[row('repeat','Un text.')],'dev':[],'cal':[]},history)
        self.assertEqual(len(cleaned['train']),1)
        with self.assertRaisesRegex(ValueError,'training history'):
            isolate_partitions({'train':[row('changed','Un text.','1')],'dev':[],'cal':[]},history)
