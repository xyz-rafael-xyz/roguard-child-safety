"""Unlaunched prototype: hash-registered original-v9 features for reviewed probes.

No fitting, label accuracy summaries, reserved TEST, or checkpoint selection.
Requires a separately reviewed root-authored plan and its exact SHA256.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import shutil
import time

FORMAT = 'roguard-candidate-original-v9-extraction-plan-1'
BASE_REVISION = '475b9088d29754a3379866cf5aeb6b41acd313c2'
ALLOWED = {'fit', 'probe_operator', 'probe_meaning', 'probe_combination',
           'probe_both', 'counterfoil'}
V9_WEIGHTS = {
    'ro': '114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
    'uk': '0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f',
}
REQUIRED_PUBLIC_SOURCES = {
    'src/roguard/gemma4_review.py', 'src/roguard/gemma4_contract.py',
    'src/roguard/gemma4_attention.py', 'src/roguard/assurance.py',
    'src/roguard/evidence.py', 'src/roguard/native_fields.py',
    'training/runtime_attention_continuation_v10.py',
    'training/assertion_probe_design_v10.py',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def canonical_json(value):
    """JSON type-sensitive identity: False/0 and True/1 must never compare equal."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_proposal_types(row):
    fields = row.get('proposed_fields')
    if (not isinstance(fields, list) or len(fields) != 6
            or any(type(value) is not int or not 0 <= value < size
                   for value, size in zip(fields, (3, 3, 4, 3, 2, 2)))
            or type(row.get('proposed_gold')) is not int or row['proposed_gold'] not in (0, 1)
            or type(row.get('source_focus')) is not int or row['source_focus'] < 0):
        raise ValueError('Proposal fields, gold and focus require exact JSON integers')


def parity_group(row):
    """Resolve optional proposal metadata before any encoder work, without eager defaults."""
    meaning = row.get('meaning_family')
    if meaning is None:
        meaning = row.get('paired_meaning_id')
    operator = row.get('operator_family')
    if operator is None:
        operator = row.get('panel_pair_id')
    if operator is None:
        operator = row.get('block_id')
    if not all(isinstance(value, str) and value.strip() for value in (meaning, operator)):
        raise ValueError('Nonempty meaning/operator parity-group metadata required')
    return (row['partition'], operator, meaning, row['proposed_attributes']['assertion'])


def check_import_origins(root, plan):
    bound = {str(root / b['path']): b['sha256'] for b in plan['bindings'].values() if b['base'] == 'public'}
    # Include the entire local serving package, not only directly imported helpers.
    for path in (root / 'src/roguard').rglob('*.py'):
        if str(path) not in bound or sha(path) != bound[str(path)]:
            raise ValueError('Serving source inventory incomplete or changed: ' + str(path))
    for name, module in list(sys.modules.items()):
        if name == 'roguard' or name.startswith('roguard.'):
            relative = Path('src') / Path(*name.split('.'))
        elif name == 'training' or name.startswith('training.'):
            relative = Path(*name.split('.'))
        else:
            continue
        origin = getattr(module, '__file__', None)
        if origin is None:
            if name != 'training' or any(Path(p).resolve() != (root / 'training').resolve() for p in module.__path__):
                raise ValueError('Unexpected local namespace origin: ' + name)
            continue
        expected = root / relative
        expected = expected / '__init__.py' if expected.is_dir() else expected.with_suffix('.py')
        if Path(origin).resolve() != expected.resolve() or str(expected) not in bound or sha(expected) != bound[str(expected)]:
            raise ValueError('Imported local source is unbound or has unexpected origin: ' + name)


def load_registered(args):
    """Pure binding checks before importing any encoder or numerical package."""
    if sha(args.plan) != args.plan_sha256:
        raise ValueError('Plan digest differs')
    plan = read(args.plan)
    if (plan['format'] != FORMAT or plan['authorization'] != 'ROOT_REVIEWED_REFERENCE_EXTRACTION_ONLY'
            or plan['extractor_sha256'] != sha(__file__)
            or plan['base_revision'] != BASE_REVISION or args.base.name != BASE_REVISION
            or plan['maximum_tokens'] != 600 or plan['prefix_tokens'] != 387
            or plan['native_replay_tolerance'] != 1e-3
            or plan['runtime_score_tolerance'] != 1e-10
            or plan['training_performed'] is not False
            or plan['held_out_test_opened'] is not False
            or plan['semantic_review_complete'] is not True
            or plan['probe_accuracy_summaries_permitted'] is not False):
        raise ValueError('Unapproved extraction scope, source or tolerances')
    paths = {}
    public_sources = set()
    for name, binding in plan['bindings'].items():
        if binding['base'] not in ('public', 'private'):
            raise ValueError('Unknown binding root')
        base = args.root if binding['base'] == 'public' else args.private_root
        relative = Path(binding['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Binding path must remain inside declared root')
        path = base / relative
        if sha(path) != binding['sha256']:
            raise ValueError('Bound source/input changed: ' + name)
        paths[name] = path
        if binding['base'] == 'public':
            public_sources.add(str(relative))
    if not REQUIRED_PUBLIC_SOURCES <= public_sources:
        raise ValueError('Required implementation binding missing')
    # Semantic and technical reports are immutable evidence, not generated by this producer.
    for name in plan['required_review_bindings']:
        if name not in paths:
            raise ValueError('Review binding missing')
    if not plan['required_review_bindings']:
        raise ValueError('Independent review references required')
    check_import_origins(args.root, plan)
    from roguard.assurance import digest, policy_digest
    from roguard.evidence import ONTOLOGY, _complete_action
    from roguard.native_fields import FIELDS
    from roguard.gemma4_contract import PREFIX_TOKENS
    from training.assertion_probe_design_v10 import normalized_payload_identity
    check_import_origins(args.root, plan)
    if len(PREFIX_TOKENS) != 387 or plan['policy_sha256'] != policy_digest():
        raise ValueError('Prefix/policy drift')
    semantic = read(paths[plan['semantic_review_binding']])
    independent = read(paths[plan['independent_checks_binding']])
    reviewed_grid = read_rows(paths[plan['reviewed_grid_rows_binding']])
    reviewed_panel = read_rows(paths[plan['reviewed_panel_rows_binding']])
    if (semantic['status'] != 'NO_NEW_MATERIAL_RENDERED_PATTERN_OR_PROPOSAL_CONSISTENCY_DEFECT_FOUND'
            or semantic['mechanical_results']['grid_rows'] != len(reviewed_grid)
            or semantic['mechanical_results']['counterfoil_rows'] != len(reviewed_panel)
            or semantic['source_bindings']['grid_rows'] != sha(paths[plan['reviewed_grid_rows_binding']])
            or semantic['source_bindings']['panel_rows'] != sha(paths[plan['reviewed_panel_rows_binding']])
            or semantic['source_bindings']['independent_checks'] != sha(paths[plan['independent_checks_binding']])
            or semantic['source_bindings']['bundle'] != sha(paths[plan['review_bundle_binding']])
            or semantic['source_bindings']['repeated_control_addendum'] != sha(paths[plan['repeated_control_binding']])
            or independent['verification'] != 'PASS_STRUCTURAL_AND_PROPOSAL_CONSISTENCY_ONLY'
            or independent['rows_checked'] != plan['total_rows']
            or semantic['training_admitted'] is not False or semantic['semantic_gold_validated'] is not False):
        raise ValueError('Typed reviewed population or retained review status differs')
    reviewed = reviewed_grid + reviewed_panel
    rows = []
    source_rows = []
    for item in plan['inputs']:
        source = read_rows(paths[item['rows_binding']])
        source_rows.extend(source)
        tokens = read_rows(paths[item['token_records_binding']])
        report = read(paths[item['token_report_binding']])
        if (len(source) != item['rows'] or len(tokens) != len(source)
                or report['complete'] is not True or report['issues']
                or report['candidates_sha256'] != sha(paths[item['rows_binding']])
                or report['records_sha256'] != sha(paths[item['token_records_binding']])
                or report['maximum_allowed_tokens'] != 600
                or report['encoder_loaded'] is not False):
            raise ValueError('Token screen population, binding or decision differs')
        for row, token in zip(source, tokens):
            validate_proposal_types(row)
            parity_group(row)
            language = row['language']
            sequence = token['tokens']
            if (language not in ('ro', 'uk') or row['payload']['language'] != language
                    or row['partition'] not in ALLOWED
                    or row['training_admitted'] is not False
                    or row['independently_annotated'] is not False
                    or row['case_id'] != token['case_id']
                    or language != token['language'] or row['partition'] != token['partition']
                    or digest(row) != token['source_row_sha256']
                    or type(token['token_count']) is not int
                    or len(sequence) != token['token_count'] or not 387 < len(sequence) <= 600
                    or any(type(v) is not int or v < 0 for v in sequence)
                    or sequence[:387] != PREFIX_TOKENS
                    or digest(sequence) != token['token_sha256']):
                raise ValueError('Candidate/token identity or scope differs')
            focus = row['source_focus']
            if focus != len(row['payload']['turns']) - 1 or row['payload']['turns'][focus]['role'] != 'minor':
                raise ValueError('Final minor focus differs')
            attributes = row['proposed_attributes']
            fields = [ONTOLOGY[k].index(attributes[k]) for k in FIELDS]
            gold = int(_complete_action(attributes) != 'CONTINUE')
            if (row['proposed_fields'] != fields or type(row['proposed_gold']) is not int
                    or row['proposed_gold'] != gold or row['policy_sha256'] != policy_digest()):
                raise ValueError('Proposed field/policy encoding drift')
            # Preserve original author proposals and retain source-order evidence.
            rows.append(dict(row, tokens=sequence, token_sha256=token['token_sha256'],
                             payload_sha256=digest(row['payload']),
                             source_row_sha256=digest(row), fields=fields, gold=gold))
    if canonical_json(source_rows) != canonical_json(reviewed):
        raise ValueError('Extraction inputs do not exactly equal reviewed grid followed by panel')
    if len(rows) != plan['total_rows'] or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Global count or duplicate case identifier')
    if len({normalized_payload_identity(r['payload']) for r in rows}) != len(rows):
        raise ValueError('Duplicate role-preserving input across grid/panel')
    if len({(r['language'], r['token_sha256']) for r in rows}) != len(rows):
        raise ValueError('Duplicate runtime input across grid/panel')
    output = {}
    for language in ('ro', 'uk'):
        current = [r for r in rows if r['language'] == language]
        config = plan['languages'][language]
        order = [[r['case_id'], r['source_row_sha256'], r['payload_sha256'], r['token_sha256']] for r in current]
        if (len(current) != config['rows'] or dict(Counter(r['partition'] for r in current)) != config['partition_counts']
                or digest(order) != config['ordered_inputs_sha256']):
            raise ValueError('Registered exact language order/population differs')
        spec = read(paths[config['spec_binding']])
        weights = paths[config['weights_binding']]
        if (sha(weights) != V9_WEIGHTS[language] or spec['adapter_sha256'] != V9_WEIGHTS[language]
                or spec['language'] != language or spec['base_revision'] != BASE_REVISION
                or spec['policy_digest'] != policy_digest()):
            raise ValueError('Not the registered original-v9 teacher')
        output[language] = dict(rows=current, weights=weights, spec=spec, order_sha256=digest(order))
    check_import_origins(args.root, plan)
    return plan, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'private-root', 'base', 'plan', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing existing extraction directory')
    plan, languages = load_registered(args)
    if shutil.disk_usage(args.private_root).free < 8 * 1024 ** 3:
        raise ValueError('Insufficient disk reserve')
    # Every import capable of loading numerical/encoder code occurs after registration.
    import numpy as np
    import mlx.core as mx
    from roguard.gemma4_review import _Engine, _score
    from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens, FILES
    from roguard.gemma4_attention import capture_with_prefix, replay
    from training.runtime_attention_continuation_v10 import native_cached_features

    check_import_origins(args.root, plan)
    def disk_check():
        reserve = 8 * 1024 ** 3
        # _Engine copies its whole pinned base to the actual system temp filesystem.
        if shutil.disk_usage(tempfile.gettempdir()).free < reserve + sum(pin['bytes'] for pin in FILES.values()):
            raise ValueError('Insufficient base snapshot copy plus temp-filesystem reserve')
        if shutil.disk_usage(args.output.parent).free < reserve:
            raise ValueError('Output-filesystem reserve reached')

    def validate(array, shape):
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Feature/logit shape, float32 type or finite values differ')

    def compare(left, right, limit):
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('Invalid numerical parity inputs')
        error = float(np.max(np.abs(left - right)))
        if error > limit:
            raise ValueError('Numerical parity exceeded: ' + str(error))
        return error

    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    lock = (args.output / 'writer.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.output / 'registered-plan.json').write_bytes(args.plan.read_bytes())
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    report = dict(complete=False, plan_sha256=args.plan_sha256, source_sha256=sha(__file__),
                  languages={}, training_performed=False, held_out_test_opened=False,
                  probe_accuracy_summaries_produced=False, training_admitted=False, qualified=False)
    started = time.monotonic()

    def persist():
        report['seconds'] = time.monotonic() - started
        temporary = args.output / 'report.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output / 'report.json')

    persist()
    for language in ('ro', 'uk'):
        data = languages[language]
        rows = data['rows']
        current = dict(complete=False, rows=len(rows), checked=0,
                       ordered_inputs_sha256=data['order_sha256'],
                       maximum_native_replay_error=0., maximum_runtime_score_error=0.,
                       score_parity_groups=[])
        report['languages'][language] = current
        with (args.output / (language + '-rows.jsonl')).open('x') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        engine = model = prefix = xs = zs = None
        suffix = state = hidden = logits = direct = x = z = None
        try:
            disk_check()
            engine = _Engine(args.base, data['weights'], V9_WEIGHTS[language])
            check_import_origins(args.root, plan)
            model = engine.model
            model.eval()
            prefix = engine.make_cache(model.base)
            mx.eval(model.base.model(mx.array([PREFIX_TOKENS]), cache=prefix))
            xs = np.lib.format.open_memmap(args.output / (language + '-features.npy'),
                                          mode='w+', dtype=np.float32, shape=(len(rows), 2560))
            zs = np.lib.format.open_memmap(args.output / (language + '-logits.npy'),
                                          mode='w+', dtype=np.float32, shape=(len(rows), 19))
            groups = set()
            with (args.output / (language + '-observations.jsonl')).open('x') as observations:
                for index, row in enumerate(rows):
                    if shutil.disk_usage(args.private_root).free < 8 * 1024 ** 3:
                        raise ValueError('Disk reserve reached')
                    if classifier_tokens(engine.tokenizer, row['payload']) != row['tokens']:
                        raise ValueError('Actual engine tokenizer differs from screened tokens')
                    suffix = mx.array([row['tokens'][387:]])
                    state = capture_with_prefix(model.base, suffix, prefix, start=24)
                    mx.eval(state)
                    hidden = replay(model.base, state)[:, 0, :]
                    logits = model.heads[language](hidden)
                    direct = native_cached_features(model, suffix, prefix)
                    mx.eval(hidden, logits, direct)
                    x = np.asarray(hidden)[0].copy()
                    z = np.asarray(logits)[0].copy()
                    validate(x, (2560,))
                    validate(z, (19,))
                    native_error = compare(x, np.asarray(direct)[0], 1e-3)
                    xs[index] = x
                    zs[index] = z
                    group = parity_group(row)
                    score_error = None
                    if group not in groups:
                        actual = engine.scores([row['payload']], data['spec'])[0]
                        check_import_origins(args.root, plan)
                        score = _score(z, data['spec'])
                        score_error = compare(np.array([score]), np.array([actual]), 1e-10)
                        if (score >= data['spec']['threshold']) != (actual >= data['spec']['threshold']):
                            raise ValueError('Served threshold decision differs')
                        groups.add(group)
                        current['score_parity_groups'].append(list(group))
                        current['maximum_runtime_score_error'] = max(current['maximum_runtime_score_error'], score_error)
                    observations.write(json.dumps(dict(index=index, case_id=row['case_id'],
                        payload_sha256=row['payload_sha256'], token_sha256=row['token_sha256'],
                        native_replay_error=native_error, runtime_score_error=score_error)) + '\n')
                    current['checked'] = index + 1
                    current['maximum_native_replay_error'] = max(current['maximum_native_replay_error'], native_error)
                    suffix = state = hidden = logits = direct = x = z = None
                    mx.clear_cache()
                    if (index + 1) % 20 == 0 or index + 1 == len(rows):
                        xs.flush()
                        zs.flush()
                        observations.flush()
                        persist()
                        print(json.dumps(dict(language=language, checked=index + 1, rows=len(rows))), flush=True)
            xs.flush()
            zs.flush()
        finally:
            # A flush failure must never skip encoder release or replace the
            # original extraction failure. Retain all partial output for audit.
            active_failure = sys.exc_info()[1]
            cleanup_errors = []
            for buffer in (xs, zs):
                if buffer is not None:
                    try:
                        buffer.flush()
                    except Exception as error:
                        cleanup_errors.append(error)
            buffer = None
            xs = zs = engine = model = prefix = None
            suffix = state = hidden = logits = direct = x = z = None
            try:
                gc.collect()
            except Exception as error:
                cleanup_errors.append(error)
            try:
                mx.clear_cache()
            except Exception as error:
                cleanup_errors.append(error)
            if cleanup_errors:
                print(json.dumps({'cleanup_failures': [type(error).__name__ + ': ' + str(error)
                                                       for error in cleanup_errors]}), file=sys.stderr)
                if active_failure is None:
                    raise cleanup_errors[0]
        current['artifacts'] = {p.name: sha(p) for p in sorted(args.output.glob(language + '-*'))}
        current['complete'] = True
        persist()
    load_registered(args)
    report['complete'] = True
    report['decision'] = 'REFERENCE_EXTRACTION_ONLY_REQUIRES_INDEPENDENT_AUDIT'
    persist()


if __name__ == '__main__':
    main()
