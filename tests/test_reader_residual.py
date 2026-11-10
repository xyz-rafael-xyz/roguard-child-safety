import hashlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('Optional reader numerical tests require numpy')

from roguard import reader_residual as public
from roguard._canonical_role_reader import Reader, MODEL_SHAPES, STATS_SHAPES
from roguard._canonical_reader_residual import Residual, MODEL_SHAPES as RESIDUAL_SHAPES
from roguard.gemma4_contract import PREFIX_TOKENS


def reader():
    weights = {k: np.zeros(s, np.float32) for k, s in MODEL_SHAPES.items()}
    stats = {k: np.ones(s, np.float64) if s == () else np.zeros(s, np.float64)
             for k, s in STATS_SHAPES.items()}
    return Reader(weights, stats)


def residual():
    weights = {k: np.zeros(s, np.float32) for k, s in RESIDUAL_SHAPES.items()}
    weights['last.bias'] = np.arange(19, dtype=np.float32) / 10
    stats = dict(q_mean=np.zeros(2560, np.float64), q_scale=np.array(1., np.float64),
                 reader_mean=np.zeros(5, np.float64), reader_scale=np.array(1., np.float64))
    return Residual(weights, stats)


def payload():
    return dict(language='ro', turns=[dict(role='minor', text='Prima afirmație.'),
                                     dict(role='adult', text='A doua afirmație.'),
                                     dict(role='minor', text='Mă refer la prima.')])


def alignment(prefix):
    groups = [[], [], []]
    assignments = []
    for i, turn in enumerate(prefix['turns']):
        slot = 0 if i == len(prefix['turns']) - 1 else 1 if turn['role'] == 'minor' else 2
        index = 387 + i
        groups[slot].append(index)
        assignments.append(dict(token_index=index, turn_index=i, role=turn['role'], pool=public.POOLS[slot]))
    return dict(role_pool_names=list(public.POOLS), role_pool_positions=groups,
                eligible_content_positions=list(range(387, 387 + len(prefix['turns']))),
                assignments=assignments, classifier_token_count=391,
                classifier_token_ids=PREFIX_TOKENS + [10, 11, 12, 13],
                classifier_ids_unchanged=True, token_truncation_applied=False)


class ReaderRuntimeTests(unittest.TestCase):
    def test_canonical_sources_are_exact(self):
        for name, expected in public.CORE_HASHES.items():
            self.assertEqual(hashlib.sha256(Path(public.__file__).with_name(name).read_bytes()).hexdigest(), expected)

    def test_role_assignment_and_missing_pool(self):
        p = payload()
        a = alignment(p)
        positions, slots = public.role_slots(p, a)
        self.assertEqual(positions, [387, 388, 389])
        self.assertEqual(slots, [[2], [0], [1]])
        values = np.ones((2, 391, 512), np.float32)
        features = public.project_features(reader(), np.ones(2560, np.float32), values, p, a)
        np.testing.assert_array_equal(features, [0, 0, 0, 0, 0, 1])
        p = dict(language='ro', turns=[p['turns'][-1]])
        features = public.project_features(reader(), np.ones(2560, np.float32), values, p, alignment(p))
        np.testing.assert_array_equal(features, np.zeros(6, np.float32))
        a['assignments'][0]['pool'] = 'prior_nonminor'
        with self.assertRaisesRegex(ValueError, 'pool differs'):
            public.role_slots(payload(), a)

    def test_manifest_rejects_parent_substitution_and_qualification(self):
        value = dict(format=public.FORMAT, language='ro', qualified=False,
                     parent_adapter_sha256=public.V9['ro'], parent_spec_sha256=public.V9_SPEC['ro'],
                     arrays={k: 'a'*64 for k in public.ARRAY_ROLES}, fit_report_sha256='b'*64,
                     reader_report_sha256='c'*64)
        self.assertEqual(public.validate_manifest(value), value)
        for key, wrong in [('qualified', True), ('parent_adapter_sha256', 'd'*64),
                           ('parent_spec_sha256', public.V9_SPEC['uk'])]:
            with self.assertRaises(ValueError):
                public.validate_manifest(dict(value, **{key: wrong}))

    def test_serving_uses_original_parent_copied_prefix_and_same_call_snapshots(self):
        model = public.ReaderResidualModel.__new__(public.ReaderResidualModel)
        model.language, model.identity = 'ro', 'a'*64
        model._reader, model._residual = reader(), residual()
        seen_cache = []
        class MX:
            float32 = np.float32
            array = staticmethod(np.array)
            @staticmethod
            def eval(*args):
                pass
            @staticmethod
            def clear_cache():
                pass
        class Base:
            @staticmethod
            def model(tokens, cache):
                cache['prefix'] = 1
                return np.zeros(1)
        original = np.arange(19, dtype=np.float32)[None, :]
        engine = types.SimpleNamespace(mx=MX, make_cache=lambda base: {},
                    model=types.SimpleNamespace(base=Base(), heads={'ro': lambda q: original.copy()}),
                    tokenizer=types.SimpleNamespace(_tokenizer=type('Tokenizer', (), {
                        'is_fast': True, '__call__': lambda self: None})()))
        model._parent = types.SimpleNamespace(_engine=engine, _spec=dict(language='ro', mode='binary', threshold=.5))
        def capture(base, suffix, cache, start):
            seen_cache.append(dict(cache))
            cache['changed'] = True
            values = np.ones((1, 2, 391, 512), np.float32)
            return dict(start=24, layers=42, length=391, shared={22: (), 23: (values, values)},
                        previous_kvs=tuple(23 if (i+1) % 6 == 0 else 22 for i in range(24, 42)))
        def tokens(tokenizer, prefix):
            if prefix['turns'][-1]['text'] == 'long':
                return PREFIX_TOKENS + [1] * (1025-len(PREFIX_TOKENS))
            return alignment(prefix)['classifier_token_ids']
        long = payload()
        long['turns'][-1]['text'] = 'long'
        with patch('roguard.gemma4_contract.classifier_tokens', side_effect=tokens), \
             patch('roguard.role_alignment.align', side_effect=lambda prefix, tok: alignment(prefix)), \
             patch('roguard.gemma4_attention.capture_with_prefix', side_effect=capture), \
             patch('roguard.gemma4_attention.replay', return_value=np.ones((1, 1, 2560), np.float32)):
            detail = model.predict_details([payload(), payload(), long])
        self.assertEqual(seen_cache, [{'prefix': 1}, {'prefix': 1}])
        expected = original[0] + np.arange(19, dtype=np.float32)/10
        self.assertEqual(np.asarray(detail[0].logits, np.float32).tobytes(), expected.tobytes())
        self.assertEqual(detail[0], detail[1])
        self.assertEqual(detail[0].reader_features, (0., 0., 0., 0., 0., 1.))
        self.assertEqual(detail[0].response['decision'], 'review')
        self.assertFalse(detail[0].response['score_is_calibrated'])
        self.assertFalse(detail[0].response['permits_external_action'])
        self.assertIsNone(detail[2].logits)
        self.assertIsNone(detail[2].reader_features)
        self.assertEqual(detail[2].response['decision'], 'clarify')


if __name__ == '__main__':
    unittest.main()
