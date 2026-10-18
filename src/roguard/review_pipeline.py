"""Run an explicit native classifier, interpreter and selected fusion model."""
import hashlib
import json
from pathlib import Path

from .field_pipeline import _local_path
from .interpreter import InterpreterModel
from .native_review import NativeReviewModel, prepare_input
from .review_fusion import ReviewFusion


class ReviewFusionPipeline:
    def __init__(self, fusion, native, interpreter):
        if (native.identity != fusion._spec['native_artifact_sha256']
                or interpreter.identity != fusion._spec['interpreter_archive_sha256']):
            raise ValueError('Pipeline components differ from selected fusion model')
        self.fusion, self.native, self.interpreter = fusion, native, interpreter
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
                or spec['format'] != 'roguard-review-fusion-bundle-v1'
                or not isinstance(spec['fusion'], dict) or set(spec['fusion']) != {'path', 'sha256'}
                or not isinstance(spec['components'], dict) or set(spec['components']) != {'native', 'interpreter'}):
            raise ValueError('Unsupported native/interpreter bundle')
        native = spec['components']['native']
        interpreter = spec['components']['interpreter']
        if (not isinstance(native, dict) or set(native) != {'artifact', 'checkpoint', 'base', 'manifest'}
                or not isinstance(interpreter, dict) or set(interpreter) != {'archive', 'base'}):
            raise ValueError('Invalid bundle component assets')
        # Resolve every path before constructing either lazy model.
        left = {k: _local_path(path.parent, v) for k, v in native.items()}
        right = {k: _local_path(path.parent, v) for k, v in interpreter.items()}
        fusion = ReviewFusion(_local_path(path.parent, spec['fusion']['path']),
                              expected_artifact_sha256=spec['fusion']['sha256'])
        return cls(fusion, NativeReviewModel(left['artifact'], left['checkpoint'], left['base'], left['manifest'],
            expected_artifact_sha256=fusion._spec['native_artifact_sha256'], device=device),
            InterpreterModel(right['archive'], right['base'],
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
        interpreted = [self.interpreter.predict(payload, focus=focus) for payload, focus in zip(payloads, focuses)]
        return [self.fusion.predict(payload, direct, semantic, focus=focus)
                for payload, direct, semantic, focus in zip(payloads, native, interpreted, focuses)]
