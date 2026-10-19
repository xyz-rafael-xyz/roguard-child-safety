from training.gemma4_prompt_v9 import classifier_tokens


def test_explicit_direct_answer_mode_overrides_wrapper_default_without_gold_access():
    class Tokenizer:
        def __init__(self):self.modes=[]
        def apply_chat_template(self, messages, *, enable_thinking=True, **kwargs):
            self.modes.append(enable_thinking)
            return [7, 8] + ([99] if enable_thinking else []) + [10+int(messages[-1]['content']), 9]
    tokenizer=Tokenizer()
    row=dict(language='ro',focus=0,turns=[dict(role='minor',text='Întrebare sintetică.')])
    assert classifier_tokens(tokenizer,row,[10,11]) == [7,8]
    assert tokenizer.modes == [False,False]
