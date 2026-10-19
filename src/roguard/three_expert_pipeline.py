"""Explicit local inference through three pinned models and their tree combiner."""
import hashlib
import json
from pathlib import Path

from .field_pipeline import _local_path
from .interpreter import InterpreterModel
from .native_fields import NativeFieldsModel
from .native_review import NativeReviewModel, prepare_input
from .three_expert import ThreeExpertReview


class ThreeExpertPipeline:
    def __init__(self, fusion, native, semantic, interpreter):
        for model, key in [(native, 'native_artifact_sha256'),
                           (semantic, 'field_artifact_sha256'),
                           (interpreter, 'interpreter_archive_sha256')]:
            if model.identity != fusion._spec[key]:
                raise ValueError('Pipeline component differs from selected model')
        self.fusion, self.native, self.semantic, self.interpreter = fusion, native, semantic, interpreter
        self.identity = fusion.identity

    @classmethod
    def from_bundle(cls, path, *, expected_sha256, device='cpu'):
        path = Path(path).resolve()
        with path.open('rb') as stream:
            raw = stream.read(65_537)
        if len(raw) > 65_536 or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('Bundle identity or size differs')
        spec = json.loads(raw)
        if (not isinstance(spec, dict) or set(spec) != {'format', 'fusion', 'components'}
                or spec['format'] != 'roguard-three-expert-bundle-v1'
                or not isinstance(spec['fusion'], dict) or set(spec['fusion']) != {'path', 'sha256'}
                or not isinstance(spec['components'], dict)
                or set(spec['components']) != {'native', 'semantic', 'interpreter'}):
            raise ValueError('Unsupported three-expert bundle')
        shapes = dict(native={'artifact', 'checkpoint', 'base', 'manifest'},
                      semantic={'heads', 'artifact', 'checkpoint', 'base', 'manifest'},
                      interpreter={'archive', 'base'})
        resolved = {}
        for name, keys in shapes.items():
            value = spec['components'][name]
            if not isinstance(value, dict) or set(value) != keys:
                raise ValueError('Invalid component assets')
            resolved[name] = {k: _local_path(path.parent, v) for k, v in value.items()}
        fusion = ThreeExpertReview(_local_path(path.parent, spec['fusion']['path']),
                                  expected_artifact_sha256=spec['fusion']['sha256'])
        n, s, i = (resolved[k] for k in ('native', 'semantic', 'interpreter'))
        return cls(fusion,
                   NativeReviewModel(n['artifact'], n['checkpoint'], n['base'], n['manifest'],
                       expected_artifact_sha256=fusion._spec['native_artifact_sha256'], device=device),
                   NativeFieldsModel(s['heads'], s['artifact'], s['checkpoint'], s['base'], s['manifest'],
                       expected_heads_sha256=fusion._spec['field_artifact_sha256'], device=device),
                   InterpreterModel(i['archive'], i['base'],
                       expected_archive_sha256=fusion._spec['interpreter_archive_sha256']))

    def predict(self, payload, *, focus=None):
        return self.predict_many([payload], focuses=[focus])[0]

    def predict_many(self, payloads, *, focuses=None):
        if not isinstance(payloads, list) or not 1 <= len(payloads) <= 128:
            raise ValueError('Expected one to 128 conversations')
        if focuses is None:
            focuses = [None] * len(payloads)
        if not isinstance(focuses, list) or len(focuses) != len(payloads):
            raise ValueError('Focus count differs')
        for payload, focus in zip(payloads, focuses):
            prepare_input(payload, focus)
            if payload['language'] != self.fusion._spec['language']:
                raise ValueError('Conversation language differs from selected model')
        native = self.native.predict_many(payloads, focuses=focuses)
        if len(native) != len(payloads):
            raise ValueError('Incomplete native batch')
        semantic = self.semantic.predict_many(payloads, focuses=focuses)
        if len(semantic) != len(payloads):
            raise ValueError('Incomplete semantic batch')
        interpreted = [self.interpreter.predict(p, focus=f) for p, f in zip(payloads, focuses)]
        return [self.fusion.predict(p, n, s, i, focus=f)
                for p, n, s, i, f in zip(payloads, native, semantic, interpreted, focuses)]
