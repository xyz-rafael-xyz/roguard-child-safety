import json
import pytest
from training.qwen_review_v9 import POLICY
from training.evaluate_qwen_adapter_v9 import POLICY as INFERENCE_POLICY
from training.train_qwen_adapter_v9 import Dataset


class Tokenizer:
    def __init__(self):self.messages=None
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs==dict(tokenize=False,add_generation_prompt=True,enable_thinking=False)
        self.messages=messages
        return messages[0]['content']+messages[1]['content']
    def encode(self,text,**kwargs):return list(map(ord,text))


def test_training_prompt_excludes_future_turns_and_targets_one_digit(tmp_path):
    assert POLICY==INFERENCE_POLICY
    row=dict(language='uk',turns=[dict(role='minor',text='Привіт.'),dict(role='minor',text='FUTURE_ONLY')],focus=0,
        attributes=dict(subject='SELF',time='CURRENT',assertion='AFFIRMED',source='DIRECT',relation='NEW',concern='0'))
    path=tmp_path/'rows.jsonl';path.write_text(json.dumps(row)+'\n');tok=Tokenizer();data=Dataset(path,tok)
    tokens,start=data.process(row)
    payload=json.loads(tok.messages[1]['content'])
    assert payload['prior_turns']==[] and payload['focal_message']==row['turns'][0]
    assert 'FUTURE_ONLY' not in tok.messages[1]['content']
    assert tokens[start:]==[ord('0')] and start==len(tokens)-1
