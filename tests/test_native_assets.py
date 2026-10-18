import pytest
from roguard.native_assets import reject_native_overrides


def test_optional_fast_tokenizer_cannot_override_the_registered_vocabulary(tmp_path):
    (tmp_path/'vocab.txt').write_text('pinned vocabulary')
    (tmp_path/'README.md').write_text('Harmless metadata')
    reject_native_overrides(tmp_path, {'config.json', 'vocab.txt'})
    (tmp_path/'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError, match='Unregistered native tokenizer'):
        reject_native_overrides(tmp_path, {'config.json', 'vocab.txt'})


def test_adapter_override_is_rejected_even_if_listed_in_manifest(tmp_path):
    (tmp_path/'adapter_config.json').write_text('{}')
    with pytest.raises(ValueError, match='Unexpected adapter'):
        reject_native_overrides(tmp_path, {'adapter_config.json'})


def test_native_manifest_cannot_name_a_parent_path(tmp_path):
    with pytest.raises(ValueError, match='plain filenames'):
        reject_native_overrides(tmp_path, {'../config.json'})
