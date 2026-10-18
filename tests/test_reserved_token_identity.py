from training.audit_reserved_tokens_v9 import token_identity


def test_shared_interpreter_cannot_hide_cross_language_token_duplicates():
    assert token_identity([1,2,3],'ro','interpreter') == token_identity([1,2,3],'uk','interpreter')
    assert token_identity([1,2,3],'ro','interpreter') != token_identity([1,2,4],'ro','interpreter')


def test_separate_native_models_keep_language_scopes():
    assert token_identity({'input_ids':[1,2]},'ro','ro') != token_identity({'input_ids':[1,2]},'uk','uk')
