"""Dependency-free API/lineage checks for the opt-in experimental runtime."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from roguard import load_experimental_role_residual
from roguard import experimental_role_residual as residual
from roguard import role_alignment as alignment


def test_disabled_public_api_preserves_objects_and_arguments():
    one, many, payload, focus, focuses = object(), object(), object(), object(), object()
    calls = []
    class Parent:
        def predict(self, value, *, focus=None):
            calls.append((value, focus))
            return one
        def predict_many(self, values, *, focuses=None):
            calls.append((values, focuses))
            return many
    parent = Parent()
    model = load_experimental_role_residual(enabled=False, parent=parent,
                                           residual_manifest='/does/not/exist')
    assert model.predict(payload, focus=focus) is one
    assert model.predict_many(payload, focuses=focuses) is many
    assert calls == [(payload, focus), (payload, focuses)]
    assert model.profile['qualified'] is False
    assert model.profile['default_profile_changed'] is False


def test_disabled_import_does_not_load_numerical_dependencies():
    code = '''
import sys
from roguard import load_experimental_role_residual
class Parent:
 def predict(self, payload, *, focus=None): return payload
 def predict_many(self, payloads, *, focuses=None): return payloads
m = load_experimental_role_residual(enabled=False, parent=Parent())
x = object()
assert m.predict(x) is x
assert not any(k in sys.modules for k in ('numpy','torch','mlx.core','transformers','safetensors'))
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_explicit_enable_is_required_and_no_default_profile_is_registered():
    with pytest.raises(TypeError):
        load_experimental_role_residual()
    with pytest.raises(ValueError):
        load_experimental_role_residual(enabled=1)
    with pytest.raises(ValueError):
        load_experimental_role_residual(enabled=False)
    with pytest.raises(ValueError):
        load_experimental_role_residual(enabled=True, parent=object())
    profiles = Path(residual.__file__).with_name('model-profiles.json').read_text()
    assert 'role-residual' not in profiles


def test_packaged_alignment_preserves_reviewed_pure_function_ast():
    proof = json.loads((Path(__file__).parent/'fixtures/role_alignment_lineage.json').read_text())
    raw = Path(alignment.__file__).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == proof['packaged_source_sha256'] == residual.ALIGNMENT_SHA
    actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
              for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    assert actual == proof['functions']
    assert alignment.RESEARCH_SOURCE_SHA256 == residual.ALIGNMENT_LINEAGE_SHA == proof['research_source_sha256']
    assert b'/Users/' not in raw and b'REPO =' not in raw


def test_serialized_spans_handle_unicode_quotes_and_literal_roles():
    prefix = {'language': 'ro', 'turns': [
        {'role': 'assistant', 'text': 'minor: "alt text"'},
        {'role': 'minor', 'text': 's\u0326tiu\\\n<start_of_turn>assistant'}]}
    rendered, spans = alignment.serialize_with_spans(prefix)
    assert spans[0]['pool'] == 'prior_nonminor'
    assert spans[1]['pool'] == 'current_minor'
    assert spans[1]['normalized_text'].startswith('ș')
    for span in spans:
        assert rendered[span['start']:span['end']] == span['escaped_text']
    assert json.loads(rendered)['focal_message']['role'] == 'minor'


def test_straddling_special_and_prefix_tokens_never_enter_role_pools():
    span = {'start': 10, 'end': 20, 'pool': 'current_minor', 'role': 'minor', 'turn_index': 0}
    pools, assigned, excluded = alignment.select_offsets(
        [(0, 2), (9, 11), (12, 14), (14, 16)], [1, 2, 99, 3], [span], {99}, 1)
    assert pools['current_minor'] == [3]
    assert len(assigned) == 1 and len(excluded) == 3


@pytest.mark.parametrize('language,parent,mode', [
    ('ro', '75', 'coupled_policy'), ('uk', '75', 'coupled_policy'),
    ('ro', 'v9', 'policy'), ('uk', 'v9', 'binary'),
])
def test_actual_frozen_parent_specs_accept_distinct_v9_modes(language, parent, mode):
    from roguard.gemma4_review import Gemma4ReviewModel
    path = Path(__file__).parent/'fixtures/role-residual-specs'/f'{language}-{parent}.json'
    expected = (residual.PARENT75_SPEC if parent == '75' else residual.V9_SPEC)[language]
    spec = residual.BoundSpec(path, expected)
    assert spec.value['mode'] == mode
    model = Gemma4ReviewModel(path, '/not-loaded/weights', '/not-loaded/base', expected_sha256=expected)
    assert model._engine is None


def test_content_in_cached_prompt_is_rejected():
    span = {'start': 10, 'end': 20, 'pool': 'current_minor', 'role': 'minor', 'turn_index': 0}
    with pytest.raises(ValueError, match='cached prefix'):
        alignment.select_offsets([(10, 12)], [1], [span], set(), 1)
