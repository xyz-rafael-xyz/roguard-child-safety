import hashlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('Optional reader numerical tests require numpy')

from roguard import focal_reader_residual as public
from roguard._canonical_role_reader import Reader, MODEL_SHAPES, STATS_SHAPES
from roguard._canonical_focal_reader_residual import Residual, MODEL_SHAPES as RESIDUAL_SHAPES
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
                 reader_mean=np.zeros(5, np.float64), reader_scale=np.array(1., np.float64),
                 focal_mean=np.zeros(1024, np.float64), focal_scale=np.array(1., np.float64))
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


class FocalReaderRuntimeTests(unittest.TestCase):
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
                     reader_report_sha256='c'*64, feature_contract=dict(public.FEATURE_CONTRACT))
        self.assertEqual(public.validate_manifest(value), value)
        for key, wrong in [('qualified', True), ('parent_adapter_sha256', 'd'*64),
                           ('parent_spec_sha256', public.V9_SPEC['uk'])]:
            with self.assertRaises(ValueError):
                public.validate_manifest(dict(value, **{key: wrong}))

    def test_serving_uses_original_parent_copied_prefix_and_same_call_snapshots(self):
        model = public.FocalReaderResidualModel.__new__(public.FocalReaderResidualModel)
        model.language, model.identity = 'ro', 'a'*64
        model._reader, model._residual = reader(), residual()
        seen_cache = []
        class MX:
            float32 = np.float32
            array = staticmethod(np.array)
            mean = staticmethod(np.mean)
            zeros = staticmethod(np.zeros)
            clear_count = 0
            fail_cleanup = False
            @staticmethod
            def eval(*args):
                pass
            @classmethod
            def clear_cache(cls):
                cls.clear_count += 1
                if cls.fail_cleanup:
                    raise ValueError('cleanup failed')
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
            before = MX.clear_count
            MX.fail_cleanup = True
            with patch('roguard.gemma4_attention.capture_with_prefix', side_effect=RuntimeError('capture failed')):
                with self.assertRaisesRegex(RuntimeError, 'capture failed') as error:
                    model.predict_details([payload()])
            self.assertEqual(MX.clear_count, before + 1)
            self.assertTrue(any('cleanup also failed' in note for note in error.exception.__notes__))
            MX.fail_cleanup = False
        self.assertEqual(seen_cache, [{'prefix': 1}, {'prefix': 1}])
        expected = original[0] + np.arange(19, dtype=np.float32)/10
        self.assertEqual(np.asarray(detail[0].logits, np.float32).tobytes(), expected.tobytes())
        self.assertEqual(detail[0], detail[1])
        self.assertEqual(detail[0].reader_features, (0., 0., 0., 0., 0., 1.))
        self.assertEqual(detail[0].focal_mean, (1.,)*1024)
        self.assertIs(detail[0].focal_present, True)
        self.assertEqual(detail[0].response['decision'], 'review')
        self.assertFalse(detail[0].response['score_is_calibrated'])
        self.assertFalse(detail[0].response['permits_external_action'])
        self.assertIsNone(detail[2].logits)
        self.assertIsNone(detail[2].reader_features)
        self.assertIsNone(detail[2].focal_mean)
        self.assertIsNone(detail[2].focal_present)
        self.assertEqual(detail[2].response['decision'], 'clarify')

    def test_native_focal_pool_uses_exact_staged_device_reduction(self):
        class MX:
            float32 = np.float32
            array = staticmethod(np.array)
            zeros = staticmethod(np.zeros)
            calls = []
            @classmethod
            def mean(cls, value, axis):
                cls.calls.append(('mean', value.shape, axis))
                return np.mean(value, axis=axis)
            @classmethod
            def eval(cls, value):
                cls.calls.append(('eval', value.shape))
        values = np.arange(1*2*391*512, dtype=np.float32).reshape(1, 2, 391, 512)
        a = alignment(payload())
        a['role_pool_positions'][0] = [388, 389]
        got, present = public.native_focal_mean(MX, values, a)
        expected = np.mean(values.transpose(0, 2, 1, 3)[0][[388, 389]], axis=0).reshape(1024)
        self.assertEqual(got.tobytes(), expected.tobytes())
        self.assertIs(present, True)
        self.assertEqual(MX.calls, [('mean', (2, 2, 512), 0), ('eval', (1024,))])
        a['role_pool_positions'][0] = []
        got, present = public.native_focal_mean(MX, values, a)
        self.assertEqual(got.tobytes(), np.zeros(1024, np.float32).tobytes())
        self.assertIs(present, False)
        for bad in ([386], [391], [389, 388], [388, 388]):
            a['role_pool_positions'][0] = bad
            with self.assertRaises(ValueError):
                public.native_focal_mean(MX, values, a)
        a['role_pool_positions'][0] = [388]
        with self.assertRaises(ValueError):
            public.native_focal_mean(MX, values.astype(np.float64), a)

    def test_absent_focal_does_not_disable_q_or_reader(self):
        model = residual()
        q = np.ones(2560, np.float32)
        reader_features = np.array([1, 2, 3, 4, 5, 1], np.float32)
        zero = np.zeros(1024, np.float32)
        prepared = model.conditioner.prepare(q, reader_features, zero, False)
        self.assertEqual(prepared.shape, (3590,))
        self.assertTrue(np.all(prepared[:2560] == 1))
        np.testing.assert_array_equal(prepared[2560:2566], reader_features)
        self.assertFalse(np.count_nonzero(prepared[2566:]))
        parent = np.zeros(19, np.float32)
        got = model.predict(q, reader_features, parent, focal=zero, focal_present=False)
        np.testing.assert_array_equal(got, np.arange(19, dtype=np.float32)/10)
        with self.assertRaises(ValueError):
            model.predict(q, reader_features, parent, focal=np.ones(1024, np.float32), focal_present=False)
        with self.assertRaises(ValueError):
            model.predict(q, reader_features, parent, focal=zero, focal_present=0)
        # Input conditioning applies no training/export gains a second time.
        np.testing.assert_array_equal(model.conditioner.prepare(q, reader_features,
            np.ones(1024, np.float32), True)[2566:], np.ones(1024, np.float32))

    def test_zero_output_and_disabled_preserve_parent_bits(self):
        model = residual()
        weights = {k: v.copy() for k, v in model.weights.items()}
        weights['last.bias'][:] = 0
        model = Residual(weights, model.conditioner.stats)
        parent = np.arange(19, dtype=np.float32)
        parent[0] = np.float32(-0.)
        kwargs = dict(focal=np.zeros(1024, np.float32), focal_present=False)
        q = np.ones(2560, np.float32)
        r = np.zeros(6, np.float32)
        self.assertEqual(model.predict(q, r, parent, **kwargs).tobytes(), parent.tobytes())
        self.assertEqual(residual().predict(q, r, parent, enabled=False, **kwargs).tobytes(), parent.tobytes())
        wrong = dict(weights, **{'first.weight': np.zeros((128, 2566), np.float32)})
        with self.assertRaises(ValueError):
            Residual(wrong, model.conditioner.stats)

    def test_whole_value_nonfinite_outside_focal_is_rejected(self):
        values = np.ones((2, 391, 512), np.float32)
        values[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Complete finite'):
            public.project_features(reader(), np.ones(2560, np.float32), values, payload(), alignment(payload()))

    def test_wrong_manifest_rejected_before_parent_or_engine_constructor(self):
        import tempfile, json
        value = dict(format=public.FORMAT, language='ro', qualified=False,
                     parent_adapter_sha256=public.V9['ro'], parent_spec_sha256=public.V9_SPEC['ro'],
                     arrays={k: 'a'*64 for k in public.ARRAY_ROLES}, fit_report_sha256='b'*64,
                     reader_report_sha256='c'*64, feature_contract=dict(public.FEATURE_CONTRACT))
        value['feature_contract']['focal_width'] = 1024.0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manifest.json'
            path.write_text(json.dumps(value))
            with patch('roguard.gemma4_review.Gemma4ReviewModel', side_effect=AssertionError('No parent on invalid manifest')):
                with self.assertRaisesRegex(ValueError, '3590-feature'):
                    public.FocalReaderResidualModel(manifest=path,
                        expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        base='unused', parent_adapter='unused', parent_spec='unused',
                        reader_weights='unused', reader_stats='unused', residual_weights='unused', residual_stats='unused')


    def test_assets_validated_before_parent_and_snapshots_are_immutable(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays = dict(reader_weights=reader().model, reader_stats=reader().stats,
                          residual_weights=residual().weights, residual_stats=residual().conditioner.stats)
            paths = {}
            for role, values in arrays.items():
                path = root/(role+'.npz')
                np.savez(path, **values)
                paths[role] = path
            value = dict(format=public.FORMAT, language='ro', qualified=False,
                parent_adapter_sha256=public.V9['ro'], parent_spec_sha256=public.V9_SPEC['ro'],
                arrays={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                fit_report_sha256='b'*64, reader_report_sha256='c'*64,
                feature_contract=dict(public.FEATURE_CONTRACT))
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(value))
            args = dict(manifest=manifest, expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                base='unused', parent_adapter='unused', parent_spec='unused', **paths)
            parent = types.SimpleNamespace(language='ro', _spec={'adapter_sha256': public.V9['ro']}, _engine=None)
            with patch('roguard.gemma4_review.Gemma4ReviewModel', return_value=parent) as constructor:
                model = public.FocalReaderResidualModel(**args)
                self.assertEqual(constructor.call_count, 1)
                self.assertEqual(model._residual.weights['first.weight'].shape, (128,3590))
                self.assertFalse(model._residual.weights['first.weight'].flags.writeable)
                copy = model.manifest
                copy['feature_contract']['focal_width'] = 9
                self.assertEqual(model.manifest['feature_contract']['focal_width'], 1024)
            # Authenticating a malformed asset's changed bytes cannot bypass its schema.
            bad = dict(arrays['residual_weights'])
            bad['first.weight'] = np.zeros((128,2566),np.float32)
            np.savez(paths['residual_weights'], **bad)
            value['arrays']['residual_weights'] = hashlib.sha256(paths['residual_weights'].read_bytes()).hexdigest()
            manifest.write_text(json.dumps(value))
            args['expected_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with patch('roguard.gemma4_review.Gemma4ReviewModel', side_effect=AssertionError('No parent on malformed tensors')):
                with self.assertRaisesRegex(ValueError, 'shape'):
                    public.FocalReaderResidualModel(**args)



if __name__ == '__main__':
    unittest.main()
