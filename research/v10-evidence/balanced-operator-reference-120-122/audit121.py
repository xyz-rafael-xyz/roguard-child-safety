"""Unlaunched independent CPU audit of the 3080-row candidate extraction.

Re-tokenizes each input and replays original-v9 heads only. Never loads an encoder.
Native/serving producer observations are recounted, not independently inferred.
"""
import argparse
from collections import Counter
import hashlib
import importlib
from importlib.metadata import version
import json
import math
import os
import re
import sys
from pathlib import Path
import tempfile
import unicodedata

for _thread_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_key] = '2'

V9 = {'ro': '114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
      'uk': '0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
COUNTS = {'fit': 288, 'probe_operator': 384, 'probe_meaning': 384,
          'probe_combination': 96, 'probe_both': 384, 'counterfoil': 4}
TOKENIZER_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
EXTRACTOR_SHA256 = '4476f0862bcd3d579e70f738bf6cf5a2eec978454aa3a7a04a8070166bb24de1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def canonical_json(value):
    """Type-sensitive canonical identity; bool/int or float/int are not interchangeable."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def exact_fields(values):
    return (isinstance(values, list) and len(values) == 6
            and all(type(value) is int and 0 <= value < size
                    for value, size in zip(values, (3, 3, 4, 3, 2, 2))))


def validate_proposal_types(row):
    if (not exact_fields(row.get('proposed_fields'))
            or type(row.get('proposed_gold')) is not int or row['proposed_gold'] not in (0, 1)
            or type(row.get('source_focus')) is not int or row['source_focus'] < 0):
        raise ValueError('Proposal fields, gold and focus require exact JSON integers')
    if (row['source_focus'] != len(row['payload']['turns']) - 1
            or row['payload']['turns'][-1]['role'] != 'minor'):
        raise ValueError('Final minor focus differs')


def finite_error(value, limit):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
        raise ValueError('Invalid parity observation')
    return float(value)


def parity_group(row):
    """Match extractor v4 without eager optional-field defaults."""
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
    return [row['partition'], operator, meaning, row['proposed_attributes']['assertion']]


def normalized_payload(payload):
    norm = lambda value: re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value).casefold()).strip()
    return [payload['language'], [(t['role'], norm(t['text'])) for t in payload['turns']]]


def check_panel_aliases(source_rows, panel, mapping, *, source_sha256, panel_sha256):
    if (len(source_rows) != 3080 or len(panel) != 40 or mapping['panel_rows'] != 40
            or mapping['canonical_rows'] != 3080 or mapping['bank_rows'] != 3072
            or mapping['bank_aliases'] != 32 or mapping['supplemental_nonfitting_rows'] != 8
            or mapping['canonical_rows_sha256'] != source_sha256
            or mapping['full_panel_sha256'] != panel_sha256
            or mapping['training_admitted'] is not False or len(mapping['aliases']) != 40):
        raise ValueError('Full40 panel alias population/binding differs')
    kinds = Counter(); identifiers = set(); slots = set()
    for index, (original, alias) in enumerate(zip(panel, mapping['aliases'])):
        slot = alias['canonical_index']
        if type(slot) is not int or not 0 <= slot < len(source_rows):
            raise ValueError('Panel alias index invalid')
        row = source_rows[slot]
        if (type(alias['panel_index']) is not int or alias['panel_index'] != index
                or not exact_fields(alias['proposed_fields'])
                or type(alias['proposed_gold']) is not int or alias['proposed_gold'] not in (0, 1)
                or type(original['proposed_binary']) is not int or original['proposed_binary'] not in (0, 1)
                or alias['panel_example_id'] != original['example_id']
                or alias['panel_source_row_sha256'] != digest(original)
                or alias['canonical_case_id'] != row['case_id'] or alias['canonical_partition'] != row['partition']
                or canonical_json(alias['proposed_fields']) != canonical_json(row['proposed_fields'])
                or canonical_json(alias['proposed_gold']) != canonical_json(row['proposed_gold'])
                or alias['labels_checked'] is not True
                or alias['normalized_payload_sha256'] != digest(normalized_payload(original['payload']))
                or normalized_payload(original['payload']) != normalized_payload(row['payload'])
                or canonical_json(original['proposed_attributes']) != canonical_json(row['proposed_attributes'])
                or original['proposed_binary'] != row['proposed_gold']
                or original['training_admitted'] is not False or original['independently_annotated'] is not False):
            raise ValueError('Full40 panel alias identity/label differs')
        if slot < 3072:
            if alias['alias_kind'] != 'EXACT_ROLE_PRESERVING_NORMALIZED_BANK_ALIAS' or row['partition'] == 'counterfoil':
                raise ValueError('Repeated panel control scope differs')
        elif (alias['alias_kind'] != 'UNIQUE_SUPPLEMENTAL_NONFITTING_CONTROL'
                or row['partition'] != 'counterfoil'
                or canonical_json(row['source_panel_row']) != canonical_json(original)):
            raise ValueError('Supplemental panel control scope differs')
        kinds[alias['alias_kind']] += 1; identifiers.add(original['example_id']); slots.add(slot)
    if (len(identifiers) != 40 or len(slots) != 40
            or kinds != {'EXACT_ROLE_PRESERVING_NORMALIZED_BANK_ALIAS': 32,
                         'UNIQUE_SUPPLEMENTAL_NONFITTING_CONTROL': 8}):
        raise ValueError('Full40 panel alias denominator differs')
    return dict(full_panel_rows=40,main_bank_alias_rows=32,unique_supplemental_rows=8,
                all_panel_responses_represented=True,independent_extra_panel_examples=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private-root', 'base', 'audit-plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--audit-plan-sha256', required=True)
    a = parser.parse_args()
    if a.output.exists() or sha(a.audit_plan) != a.audit_plan_sha256:
        raise ValueError('Audit output exists or audit plan differs')
    plan = read(a.audit_plan)
    if (plan['format'] != 'roguard-balanced-operator-features-audit-plan-1'
            or plan['source_sha256'] != sha(__file__) or plan['cpu_logit_tolerance'] != 1e-3
            or plan['authorization'] != 'ROOT_REVIEWED_CPU_FEATURE_AUDIT_ONLY'):
        raise ValueError('Unsupported audit registration')
    snapshot = {a.audit_plan: a.audit_plan_sha256, Path(__file__): sha(__file__)}

    def bind(base, relative, expected):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Binding escapes declared root')
        path = base / relative
        if sha(path) != expected:
            raise ValueError('Binding changed: ' + str(path))
        snapshot[path] = expected
        return path

    audit_paths = {}
    for key, item in plan['bindings'].items():
        if item['base'] not in ('public', 'private'):
            raise ValueError('Unknown binding root')
        audit_paths[key] = bind(a.root if item['base'] == 'public' else a.private_root,
                                item['path'], item['sha256'])
    extraction_path = audit_paths[plan['extraction_plan_binding']]
    extraction = read(extraction_path)
    data = a.private_root / plan['extraction_output']
    if not data.resolve().is_relative_to(a.private_root.resolve()):
        raise ValueError('Output escapes private root')
    producer_path = audit_paths[plan['producer_report_binding']]
    if producer_path.resolve() != (data / 'report.json').resolve():
        raise ValueError('Wrong producer report')
    producer = read(producer_path)
    if (extraction['format'] != 'roguard-candidate-original-v9-extraction-plan-1'
            or extraction['authorization'] != 'ROOT_REVIEWED_REFERENCE_EXTRACTION_ONLY'
            or extraction['total_rows'] != 3080 or extraction['maximum_tokens'] != 600
            or extraction['prefix_tokens'] != 387 or extraction['semantic_review_complete'] is not True
            or extraction['extractor_sha256'] != EXTRACTOR_SHA256
            or extraction['native_replay_tolerance'] != 1e-3 or extraction['runtime_score_tolerance'] != 1e-10
            or producer['complete'] is not True or set(producer['languages']) != {'ro', 'uk'}
            or producer['decision'] != 'REFERENCE_EXTRACTION_ONLY_REQUIRES_INDEPENDENT_AUDIT'
            or producer['plan_sha256'] != sha(extraction_path)
            or producer['source_sha256'] != extraction['extractor_sha256']
            or any(producer[key] is not False for key in ('training_performed', 'held_out_test_opened',
                'probe_accuracy_summaries_produced', 'training_admitted', 'qualified'))):
        raise ValueError('Incomplete producer or wrong scope')
    bind(data, 'registered-plan.json', sha(extraction_path))
    bind(data, 'source.py', extraction['extractor_sha256'])
    input_paths = {}
    for key, item in extraction['bindings'].items():
        if item['base'] not in ('public', 'private'):
            raise ValueError('Unknown extractor binding root')
        input_paths[key] = bind(a.root if item['base'] == 'public' else a.private_root,
                               item['path'], item['sha256'])
    semantic = read(input_paths[extraction['semantic_review_binding']])
    independent = read(input_paths[extraction['independent_checks_binding']])
    reviewed_grid = rows(input_paths[extraction['reviewed_grid_rows_binding']])
    reviewed_panel = rows(input_paths[extraction['reviewed_panel_rows_binding']])
    if (len(reviewed_grid) != 3072 or len(reviewed_panel) != 8
            or semantic['status'] != 'NO_NEW_MATERIAL_RENDERED_PATTERN_OR_PROPOSAL_CONSISTENCY_DEFECT_FOUND'
            or semantic['mechanical_results']['grid_rows'] != 3072
            or semantic['mechanical_results']['counterfoil_rows'] != 8
            or semantic['source_bindings']['grid_rows'] != sha(input_paths[extraction['reviewed_grid_rows_binding']])
            or semantic['source_bindings']['panel_rows'] != sha(input_paths[extraction['reviewed_panel_rows_binding']])
            or semantic['source_bindings']['independent_checks'] != sha(input_paths[extraction['independent_checks_binding']])
            or semantic['source_bindings']['bundle'] != sha(input_paths[extraction['review_bundle_binding']])
            or semantic['source_bindings']['repeated_control_addendum'] != sha(input_paths[extraction['repeated_control_binding']])
            or independent['verification'] != 'PASS_STRUCTURAL_AND_PROPOSAL_CONSISTENCY_ONLY'
            or independent['rows_checked'] != 3080
            or semantic['training_admitted'] is not False or semantic['semantic_gold_validated'] is not False):
        raise ValueError('Typed reviewed3080 population differs')
    def check_local_origins():
        for path in (a.root / 'src/roguard').rglob('*.py'):
            if path not in snapshot or sha(path) != snapshot[path]:
                raise ValueError('Complete serving source inventory is not bound: ' + str(path))
        for name, module in list(sys.modules.items()):
            if name == 'roguard' or name.startswith('roguard.'):
                relative = Path('src') / Path(*name.split('.'))
            elif name == 'training' or name.startswith('training.'):
                relative = Path(*name.split('.'))
            else:
                continue
            origin = getattr(module, '__file__', None)
            if origin is None:
                if name != 'training' or any(Path(p).resolve() != (a.root / 'training').resolve() for p in module.__path__):
                    raise ValueError('Unexpected local namespace origin: ' + name)
                continue
            expected = a.root / relative
            expected = expected / '__init__.py' if expected.is_dir() else expected.with_suffix('.py')
            if Path(origin).resolve() != expected.resolve() or expected not in snapshot or sha(expected) != snapshot[expected]:
                raise ValueError('Imported local source is unbound or unexpected: ' + name)
    check_local_origins()
    # Audit explicitly rejects unexpected import origins rather than trusting PYTHONPATH.
    for module, relative in {
        'roguard.assurance': 'src/roguard/assurance.py',
        'roguard.evidence': 'src/roguard/evidence.py',
        'roguard.native_fields': 'src/roguard/native_fields.py',
        'roguard.gemma4_contract': 'src/roguard/gemma4_contract.py',
        'roguard.gemma4_review': 'src/roguard/gemma4_review.py',
        'training.shared_tail_training_v9': 'training/shared_tail_training_v9.py',
    }.items():
        path = a.root / relative
        if path not in snapshot or Path(importlib.import_module(module).__file__).resolve() != path.resolve():
            raise ValueError('Audit import is unbound or resolves outside reviewed source: ' + module)
    from roguard.gemma4_contract import FILES, VERSIONS, PREFIX_TOKENS, classifier_tokens
    from roguard.assurance import policy_digest
    from roguard.evidence import ONTOLOGY, _complete_action
    from roguard.native_fields import FIELDS
    from roguard.gemma4_review import _score
    check_local_origins()
    if len(PREFIX_TOKENS) != 387 or extraction['policy_sha256'] != policy_digest():
        raise ValueError('Prefix/policy changed')
    expected = []; source_rows = []
    for item in extraction['inputs']:
        original = rows(input_paths[item['rows_binding']])
        source_rows.extend(original)
        token_rows = rows(input_paths[item['token_records_binding']])
        screen = read(input_paths[item['token_report_binding']])
        if (len(original) != item['rows'] or len(token_rows) != len(original)
                or screen['complete'] is not True or screen['issues']
                or screen['candidates_sha256'] != sha(input_paths[item['rows_binding']])
                or screen['records_sha256'] != sha(input_paths[item['token_records_binding']])):
            raise ValueError('Screen source or population drift')
        for original_row, token in zip(original, token_rows):
            validate_proposal_types(original_row)
            parity_group(original_row)
            if (token['case_id'] != original_row['case_id']
                    or token['language'] != original_row['language']
                    or token['partition'] != original_row['partition']
                    or token['source_row_sha256'] != digest(original_row)
                    or token['token_sha256'] != digest(token['tokens'])
                    or token['token_count'] != len(token['tokens'])
                    or type(token['token_count']) is not int
                    or any(type(value) is not int or value < 0 for value in token['tokens'])
                    or not 387 < len(token['tokens']) <= 600
                    or token['tokens'][:387] != PREFIX_TOKENS):
                raise ValueError('Token record order/source differs')
            attrs = original_row['proposed_attributes']
            fields = [ONTOLOGY[key].index(attrs[key]) for key in FIELDS]
            gold = int(_complete_action(attrs) != 'CONTINUE')
            if (original_row['proposed_fields'] != fields or type(original_row['proposed_gold']) is not int
                    or original_row['proposed_gold'] != gold or original_row['policy_sha256'] != policy_digest()
                    or original_row['training_admitted'] is not False
                    or original_row['independently_annotated'] is not False):
                raise ValueError('Reviewed label encoding changed')
            expected.append(dict(original_row, tokens=token['tokens'], token_sha256=token['token_sha256'],
                payload_sha256=digest(original_row['payload']), source_row_sha256=digest(original_row),
                fields=fields, gold=gold))
    if len(expected) != 3080 or len({row['case_id'] for row in expected}) != 3080:
        raise ValueError('Expected exact 3080 reviewed input rows')
    if canonical_json(source_rows) != canonical_json(reviewed_grid + reviewed_panel) or len(extraction['inputs']) != 1:
        raise ValueError('Canonical reviewed grid plus unique supplements differ')
    panel_path = audit_paths[plan['full_panel_binding']]
    alias_path = audit_paths[plan['panel_alias_map_binding']]
    panel_summary = check_panel_aliases(source_rows, rows(panel_path), read(alias_path),
        source_sha256=sha(input_paths[extraction['inputs'][0]['rows_binding']]), panel_sha256=sha(panel_path))
    for package in ('transformers', 'tokenizers', 'huggingface-hub', 'mlx', 'mlx-lm', 'numpy'):
        if version(package) != VERSIONS[package]:
            raise ValueError('Pinned software version differs: ' + package)
    import numpy as np
    import mlx.core as mx
    from safetensors.numpy import load_file
    from training.shared_tail_training_v9 import make_head
    from transformers import AutoTokenizer
    check_local_origins()
    mx.set_default_device(mx.cpu)
    summaries, feature_artifacts = {}, {}
    token_count = head_count = 0
    with tempfile.TemporaryDirectory(prefix='roguard-probe-token-audit-') as temporary:
        tokenizer_path = Path(temporary)
        for name in TOKENIZER_FILES:
            pin = FILES[name]
            raw = (a.base / name).read_bytes()
            if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
                raise ValueError('Tokenizer asset differs')
            snapshot[a.base / name] = pin['sha256']
            (tokenizer_path / name).write_bytes(raw)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
        for language in ('ro', 'uk'):
            wanted = [row for row in expected if row['language'] == language]
            config = extraction['languages'][language]
            record = producer['languages'][language]
            if (len(wanted) != 1540 or dict(Counter(row['partition'] for row in wanted)) != COUNTS
                    or config['rows'] != 1540 or config['partition_counts'] != COUNTS
                    or record['rows'] != 1540 or record['checked'] != 1540 or record['complete'] is not True):
                raise ValueError('Fit/probe/panel membership or producer completion differs')
            order = digest([[r['case_id'], r['source_row_sha256'], r['payload_sha256'], r['token_sha256']] for r in wanted])
            if order != config['ordered_inputs_sha256'] or order != record['ordered_inputs_sha256']:
                raise ValueError('Registered order differs')
            names = {language + '-' + suffix for suffix in ('rows.jsonl', 'features.npy', 'logits.npy', 'observations.jsonl')}
            if set(record['artifacts']) != names:
                raise ValueError('Producer artifact inventory differs')
            for name in sorted(names):
                bind(data, name, record['artifacts'][name])
                if not name.endswith('observations.jsonl'):
                    feature_artifacts[name] = record['artifacts'][name]
            actual_rows = rows(data / (language + '-rows.jsonl'))
            if canonical_json(actual_rows) != canonical_json(wanted):
                raise ValueError('Feature row order/source/labels differ')
            for row in actual_rows:
                if classifier_tokens(tokenizer, row['payload']) != row['tokens']:
                    raise ValueError('Independent full runtime token replay differs')
                token_count += 1
            x = np.load(data / (language + '-features.npy'), allow_pickle=False)
            z = np.load(data / (language + '-logits.npy'), allow_pickle=False)
            for array, shape in ((x, (1540, 2560)), (z, (1540, 19))):
                if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError('Feature/logit array invalid')
            observations = rows(data / (language + '-observations.jsonl'))
            if len(observations) != 1540:
                raise ValueError('Native observation population differs')
            groups, maximum_native, maximum_score = [], 0., 0.
            for index, (row, observation) in enumerate(zip(actual_rows, observations)):
                if (type(observation['index']) is not int or observation['index'] != index
                        or any(observation[key] != row[key] for key in ('case_id', 'payload_sha256', 'token_sha256'))):
                    raise ValueError('Native observation identity/order differs')
                maximum_native = max(maximum_native, finite_error(observation['native_replay_error'], 1e-3))
                group = parity_group(row)
                if group not in groups:
                    groups.append(group)
                    maximum_score = max(maximum_score, finite_error(observation['runtime_score_error'], 1e-10))
                elif observation['runtime_score_error'] is not None:
                    raise ValueError('Unexpected duplicate score parity observation')
            if (groups != record['score_parity_groups'] or maximum_native != finite_error(record['maximum_native_replay_error'], 1e-3)
                    or maximum_score != finite_error(record['maximum_runtime_score_error'], 1e-10)):
                raise ValueError('Producer parity summary differs from full observations')
            weights = input_paths[config['weights_binding']]
            spec = read(input_paths[config['spec_binding']])
            if sha(weights) != V9[language] or spec['adapter_sha256'] != V9[language] or spec['language'] != language:
                raise ValueError('Original-v9 teacher identity differs')
            tensors = load_file(weights)
            mapping = {'0.weight': 'first.weight', '0.bias': 'first.bias', '3.weight': 'last.weight', '3.bias': 'last.bias'}
            head = make_head({key: tensors['heads.' + language + '.' + name] for key, name in mapping.items()})
            head.eval()
            actual = head(mx.array(x))
            mx.eval(actual)
            replay = np.asarray(actual).copy()
            if replay.shape != z.shape or replay.dtype != np.float32 or not np.isfinite(replay).all():
                raise ValueError('CPU head output invalid')
            logit_error = finite_error(float(np.max(np.abs(replay - z))), 1e-3)
            def classes(values):
                return np.column_stack([part.argmax(1) for part in np.split(values, [3, 6, 10, 13, 15, 17], axis=1)])
            argmax_changes = int(np.any(classes(replay) != classes(z), axis=1).sum())
            decision_changes = sum((_score(left, spec) >= spec['threshold']) != (_score(right, spec) >= spec['threshold'])
                                   for left, right in zip(replay, z))
            if argmax_changes or decision_changes:
                raise ValueError('Exact seven-head argmax or binary decision parity failed')
            head_count += len(actual_rows)
            summaries[language] = dict(rows=1540, partitions=COUNTS, ordered_inputs_sha256=order,
                maximum_cpu_logit_error=logit_error, seven_head_argmax_rows_differing=argmax_changes,
                binary_decision_differences=decision_changes, native_observations_recounted=1540,
                producer_native_maximum_recounted=maximum_native, producer_score_maximum_recounted=maximum_score,
                producer_score_groups_recounted=len(groups), native_encoder_independently_replayed=False)
            del head, tensors, actual, replay, x, z
            mx.clear_cache()
    check_local_origins()
    if token_count != 3080 or head_count != 3080 or len(feature_artifacts) != 6:
        raise ValueError('Incomplete independent audit')
    if any(sha(path) != expected_sha for path, expected_sha in snapshot.items()):
        raise ValueError('Bound source or artifact changed during audit')
    result = dict(format='roguard-assertion-probe-features-audit-1', verification='PASS',
        source_sha256=sha(__file__), audit_plan_sha256=a.audit_plan_sha256,
        extraction_plan_sha256=sha(extraction_path), producer_report_sha256=sha(producer_path),
        feature_artifacts=feature_artifacts, encoder_weights=V9, token_rows_replayed=token_count,
        head_rows_replayed=head_count, languages=summaries, device='cpu', encoder_loaded=False,
        panel_alias_summary=panel_summary, full_panel_sha256=sha(panel_path), panel_alias_map_sha256=sha(alias_path),
        native_encoder_independently_replayed=False, probe_accuracy_summaries_produced=False,
        training_performed=False, training_admitted=False, held_out_test_opened=False, qualified=False,
        limitations='CPU token/head replay and exact row/artifact checks only. Native/serving parity is a full recount of producer observations, not independent encoder inference or semantic gold validation.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({key: result[key] for key in ('verification', 'token_rows_replayed', 'head_rows_replayed', 'qualified')}))


if __name__ == '__main__':
    main()
