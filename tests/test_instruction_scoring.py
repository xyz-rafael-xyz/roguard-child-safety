import unittest
from training.evaluate_instruction_sft_v8 import classifier_tokens


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        # A fixed empty reasoning block is inserted before the answer.
        return [90,91,92,93,int(messages[-1]['content'])+10,99]


class InstructionScoringTests(unittest.TestCase):
    def test_label_prefix_includes_template_scaffolding(self):
        row={'turns':[{'role':'minor','text':'example'}],'focus':0}
        self.assertEqual(classifier_tokens(Tokenizer(),row,[10,11]),[90,91,92,93])

    def test_gold_label_is_not_read(self):
        row={'turns':[{'role':'minor','text':'example'}],'focus':0,'attributes':{'concern':'0'}}
        first=classifier_tokens(Tokenizer(),row,[10,11]);row['attributes']['concern']='1'
        self.assertEqual(first,classifier_tokens(Tokenizer(),row,[10,11]))

    def test_wrong_label_token_is_rejected(self):
        with self.assertRaises(ValueError):
            classifier_tokens(Tokenizer(),{'turns':[{'text':'example'}],'focus':0},[0,1])

class LabelDatasetTests(unittest.TestCase):
    def test_only_class_token_is_supervised(self):
        import json
        from pathlib import Path
        import tempfile
        from training.train_instruction_labels_v8 import LabelDataset
        class TrainTokenizer(Tokenizer):
            def encode(self,text,**kwargs):return [int(text)+10]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train.jsonl'
            path.write_text(json.dumps({'messages':[{'role':'user','content':'example'}, {'role':'assistant','content':'1'}]})+'\n',encoding='utf-8')
            dataset=LabelDataset(path,TrainTokenizer())
            tokens,offset=dataset.process(dataset[0])
            self.assertEqual(tokens,[90,91,92,93,11])
            self.assertEqual(tokens[offset:],[11])
            self.assertNotIn(99,tokens)
