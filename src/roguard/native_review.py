"""Offline runtime for explicitly selected native-language review checkpoints.

This adapter does not approve a model for release. Scores are model proposals,
not calibrated risk estimates, verified disclosures or permission grants.
"""
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading

from .context import render, validate
from .native_assets import reject_native_overrides
from .semantic import normalize_semantic_text


BASES = {
    'ro': ('dumitrescustefan/bert-base-romanian-cased-v1',
           '37fb0ffb4bc4f7c4cde429626775685fb18f234f', 'bert',
           frozenset({'config.json', 'tokenizer_config.json', 'vocab.txt', 'model.safetensors'})),
    'uk': ('youscan/ukr-roberta-base',
           '8149bde480a6df9014aa934c3f8af30858dea5a6', 'roberta',
           frozenset({'config.json', 'tokenizer_config.json', 'special_tokens_map.json',
                      'vocab.json', 'merges.txt', 'pytorch_model.bin'})),
}
PREPROCESSING = ('NFC, semantic whitespace/case normalization, Romanian cedilla-to-comma; '
                 'context/focal pair; max512, no truncation.')
WEIGHT_FILES = frozenset({'model.safetensors', 'pytorch_model.bin'})


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a lowercase SHA-256 digest')
    return value


def _read(path, limit):
    with Path(path).open('rb') as source:
        data = source.read(limit+1)
    if len(data) > limit:
        raise ValueError('Native model asset exceeds its size limit')
    return data


def prepare_input(payload, focus=None):
    """Validate only the selected prefix; future content never reaches the model."""
    if not isinstance(payload, dict) or set(payload) != {'language', 'turns'}:
        raise ValueError('Expected language and turns')
    turns = payload['turns']
    if not isinstance(turns, list) or not turns:
        raise ValueError('Expected nonempty turns')
    if focus is None:
        focus = len(turns)-1
    if type(focus) is not int or not 0 <= focus < len(turns):
        raise ValueError('Invalid focal turn index')
    if focus >= 12:
        raise ValueError('Selected prefix exceeds 12 turns')
    prefix = dict(language=payload['language'], turns=turns[:focus+1])
    validate(prefix)
    # Validate the bounded, plain-data shape before copying caller objects.
    prefix = deepcopy(prefix)
    checked = prefix['turns']
    if checked[-1]['role'] != 'minor':
        raise ValueError('Focal turn must be a minor message')
    normalize = lambda value: normalize_semantic_text(value).translate(str.maketrans('şţŞŢ', 'șțȘȚ'))
    pair = (normalize(render(checked)), normalize(checked[-1]['text']))
    identity = _sha(json.dumps(prefix, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode())
    return pair, focus, identity


class _TorchEngine:
    def __init__(self, files, checkpoint, checkpoint_digest, model_type, device):
        try:
            import torch
            from safetensors.torch import load
            from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as error:
            raise ImportError('Native review requires the roguard-child-safety[semantic] extra') from error
        self.torch = torch
        self.device = device
        # Read only verified tokenizer bytes from a private snapshot. Unknown
        # local files and a changed cache directory cannot override this load.
        with tempfile.TemporaryDirectory(prefix='roguard-native-assets-') as temporary:
            path = Path(temporary)
            for name, data in files.items():
                (path/name).write_bytes(data)
            config = AutoConfig.from_pretrained(path, num_labels=2, local_files_only=True, trust_remote_code=False)
            if config.model_type != model_type:
                raise ValueError('Native model architecture differs')
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=False).float()
        raw = _read(checkpoint, 700_000_000)
        if _sha(raw) != checkpoint_digest:
            raise ValueError('Native checkpoint hash differs')
        # Deserialize the same bytes that were hashed; never load pickle or a
        # second potentially changed file after checking the first one.
        tensors = load(raw)
        del raw
        for tensor in tensors.values():
            if tensor.is_floating_point() and (tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()):
                raise ValueError('Expected finite float32 native weights')
        self.model.load_state_dict(tensors, strict=True)
        del tensors
        self.model = self.model.to(device).eval()

    def scores(self, pairs):
        encoded = self.tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                                 padding=False, truncation=False)
        valid = [i for i, ids in enumerate(encoded['input_ids']) if len(ids) <= 512]
        result = [None]*len(pairs)
        with self.torch.inference_mode():
            for offset in range(0, len(valid), 8):
                indices = valid[offset:offset+8]
                batch = self.tokenizer.pad([{key: values[i] for key, values in encoded.items()} for i in indices],
                                           return_tensors='pt')
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(**batch).logits.float()
                if tuple(logits.shape) != (len(indices), 2):
                    raise ValueError('Native classifier output shape differs')
                scores = self.torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                for i, score in zip(indices, scores):
                    result[i] = score
        return result


class NativeReviewModel:
    """An explicit, hash-bound checkpoint; local paths only, no network calls.

    Keep the expected artifact hash in trusted application configuration.
    This class neither chooses a checkpoint nor changes the package default.
    """
    def __init__(self, artifact, checkpoint, base, base_manifest, *, expected_artifact_sha256, device='cpu'):
        if device not in ('cpu', 'mps', 'cuda'):
            raise ValueError('Unsupported native model device')
        raw = _read(artifact, 65536)
        if _sha(raw) != _digest(expected_artifact_sha256):
            raise ValueError('Native review artifact hash differs')
        import io
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
                unpacked = source.read(65537)
        except (OSError, EOFError) as error:
            raise ValueError('Invalid compressed native metadata') from error
        if len(unpacked) > 65536:
            raise ValueError('Oversized native review metadata')
        spec = json.loads(unpacked)
        required = {'format', 'base', 'revision', 'language', 'thresholds',
                    'checkpoint_sha256', 'base_manifest_sha256', 'preprocessing'}
        if not isinstance(spec, dict) or set(spec) != required or spec['format'] != 'roguard-native-full-review-experiment-v1':
            raise ValueError('Unsupported native review artifact')
        language = spec['language']
        if not isinstance(language, str) or language not in BASES:
            raise ValueError('Unsupported native review language')
        model_id, revision, model_type, allowed = BASES[language]
        if (spec['base'], spec['revision'], spec['preprocessing']) != (model_id, revision, PREPROCESSING):
            raise ValueError('Native base or preprocessing identity differs')
        thresholds = spec['thresholds']
        if not isinstance(thresholds, dict) or set(thresholds) != {language}:
            raise ValueError('Native threshold language differs')
        threshold = thresholds[language]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Invalid native review threshold')
        manifest_bytes = _read(base_manifest, 65536)
        if _sha(manifest_bytes) != _digest(spec['base_manifest_sha256']):
            raise ValueError('Native base manifest hash differs')
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or not isinstance(manifest.get('models'), dict):
            raise ValueError('Invalid native base manifest')
        entry = manifest.get('models', {}).get(language, {})
        if (not isinstance(entry, dict) or entry.get('model') != model_id or entry.get('revision') != revision
                or not isinstance(entry.get('files'), dict) or set(entry['files']) != allowed):
            raise ValueError('Native base file inventory differs')
        reject_native_overrides(base, entry['files'])
        files = {}
        for name in allowed-WEIGHT_FILES:
            details = entry['files'][name]
            if not isinstance(details, dict) or type(details.get('bytes')) is not int or not 0 < details['bytes'] <= 8_000_000:
                raise ValueError('Invalid native tokenizer file size')
            data = _read(Path(base)/name, 8_000_000)
            if len(data) != details['bytes'] or _sha(data) != _digest(details['sha256']):
                raise ValueError('Native tokenizer/configuration file hash differs: '+name)
            files[name] = data
        self._language = language
        self._threshold = float(threshold)
        self._identity = expected_artifact_sha256
        self._engine = None
        self._lock = threading.RLock()
        self._engine_args = (files, Path(checkpoint), _digest(spec['checkpoint_sha256']), model_type, device)

    @property
    def language(self):
        return self._language

    @property
    def threshold(self):
        return self._threshold

    @property
    def identity(self):
        return self._identity

    def predict(self, payload, *, focus=None):
        return self.predict_many([payload], focuses=[focus])[0]

    def predict_many(self, payloads, *, focuses=None):
        if not isinstance(payloads, list) or not 1 <= len(payloads) <= 128:
            raise ValueError('Expected one to 128 conversations')
        if focuses is None:
            focuses = [None]*len(payloads)
        if not isinstance(focuses, list) or len(focuses) != len(payloads):
            raise ValueError('Focal indices do not match the batch')
        prepared = [prepare_input(payload, focus) for payload, focus in zip(payloads, focuses)]
        if any(payload['language'] != self.language for payload in payloads):
            raise ValueError('Conversation language differs from selected model')
        with self._lock:
            if self._engine is None:
                self._engine = _TorchEngine(*self._engine_args)
            scores = self._engine.scores([p[0] for p in prepared])
        if not isinstance(scores, list) or len(scores) != len(prepared):
            raise ValueError('Incomplete native model output')
        outputs = []
        for score, (_, focus, identity) in zip(scores, prepared):
            if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Invalid native model score')
            predicted = None if score is None else int(score >= self.threshold)
            outputs.append(dict(format='roguard-native-review-v1', language=self.language, focus=focus,
                input_digest=identity, model_identity=self.identity, predicted=predicted, score=score,
                decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
                reason='NATIVE_CONTEXT_LIMIT' if predicted is None else 'MODEL_REVIEW_PROPOSAL' if predicted else 'NO_SIGNAL_DETECTED',
                review_requested=predicted != 0, score_is_calibrated=False, text_in_report=False,
                establishes_safety=False, permits_external_action=False, permits_notification=False))
        return outputs
