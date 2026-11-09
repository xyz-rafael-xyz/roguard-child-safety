"""Inspect local v10 research results. No external actions or release approval."""
import argparse
import json
from pathlib import Path


_JSON_LIMIT = 4_000_000


def _read_json(path):
    with path.open('rb') as source:
        raw = source.read(_JSON_LIMIT + 1)
    if len(raw) > _JSON_LIMIT:
        raise ValueError('JSON exceeds file-size limit')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model', type=Path, help='Legacy research selector artifact')
    mode.add_argument('--bundle', type=Path, help='Explicit v10 bundle manifest')
    mode.add_argument('--profile', help='Explicit installed model profile; never downloads')
    p.add_argument('--cache', type=Path, help='Model cache directory, only with --profile')
    p.add_argument('--model-sha256')
    p.add_argument('--bundle-sha256')
    p.add_argument('--calibration', type=Path)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--neural', action='store_true', help='Use pinned v9 components with the research selector')
    p.add_argument('--draft', help='Assistant reply to review')
    p.add_argument('--context', help='Caller-declared response context; required with bundle/profile --draft, may be empty')
    a = p.parse_args(argv)
    public_mode = a.bundle is not None or a.profile is not None
    if public_mode:
        if a.model_sha256 is not None or a.neural or a.plan:
            p.error('--bundle/--profile cannot use --model-sha256, --neural or --plan')
        if a.bundle is not None and not a.bundle_sha256:
            p.error('--bundle requires --bundle-sha256')
        if a.profile is not None and a.bundle_sha256 is not None:
            p.error('--profile uses the catalog checksum; do not supply --bundle-sha256')
        if a.draft is not None and a.context is None:
            p.error('Bundle/profile --draft requires an explicit --context (use an empty string for no context)')
    elif not a.model_sha256 or a.bundle_sha256 is not None:
        p.error('--model requires --model-sha256 and cannot use --bundle-sha256')
    if a.cache is not None and a.profile is None:
        p.error('--cache requires --profile')
    if a.context is not None and (a.draft is None or not public_mode):
        p.error('--context requires --bundle/--profile and --draft')
    try:
        payload = _read_json(a.input)
        calibration = _read_json(a.calibration) if a.calibration else None
        if a.plan and calibration is None:
            raise ValueError('Planning requires a pipeline-bound calibration artifact')
        context = a.context
        if a.draft is not None:
            if payload.get('language') not in ('ro', 'uk'):
                raise ValueError('Response review requires Romanian or Ukrainian language')
            if not a.draft.strip() or len(a.draft) > 6000:
                raise ValueError('Draft must contain 1 to 6000 characters')
            if context is None:  # Preserve the explicitly selected legacy research route.
                context = payload['turns'][-1]['text']
            if not isinstance(context, str) or len(context) > 12000:
                raise ValueError('Response context must contain at most 12000 characters')
            a.draft.encode('utf-8')
            context.encode('utf-8')
        model = None
        try:
            if public_mode:
                bundle_path, bundle_sha256 = a.bundle, a.bundle_sha256
                if a.profile is not None:
                    from .model_install import cache_directory, install, profile
                    raw, manifest_sha256 = profile(a.profile)
                    bundle_path, bundle_sha256 = install(
                        raw, manifest_sha256, a.cache if a.cache is not None else cache_directory(),
                        download=False)
                from .v10_bundle import load_v10
                model = load_v10(bundle_path, expected_sha256=bundle_sha256)
                result = dict(model.predict(payload, calibration=calibration))
                if a.draft is not None:
                    result['response_review'] = model.review_response(a.draft, payload['language'], context=context)
            else:
                from .trajectory_model import TrajectoryModel
                selector = TrajectoryModel(a.model, expected_sha256=a.model_sha256)
                model = selector
                if a.neural:
                    from .trajectory_neural import NeuralTrajectoryModel
                    model = NeuralTrajectoryModel.from_profile(selector)
                result = dict(model.predict(payload, calibration=calibration, plan=a.plan))
                result.pop('probabilities', None)
                if a.draft is not None:
                    result['response_review'] = selector.review_response(a.draft, payload['language'], context)
            # Keep the model's experimental/qualification flags unchanged.
            output = json.dumps(result, indent=2, ensure_ascii=True)
        finally:
            if model is not None:
                close = getattr(model, 'close', None)
                if callable(close):
                    close()
        print(output)
    except (ValueError, KeyError, IndexError, TypeError, OSError, ImportError, RuntimeError, UnicodeError) as error:
        p.exit(2, 'Invalid v10 input: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
