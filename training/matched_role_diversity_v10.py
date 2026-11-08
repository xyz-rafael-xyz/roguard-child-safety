"""Deterministic TRAIN-only matching for a paired semantic-diversity experiment."""
from collections import Counter, defaultdict

from roguard.assurance import digest
from training.native_fields_v9 import SIZES


def label_key(row):
    fields = row['fields']
    if (not isinstance(fields, list) or len(fields) != len(SIZES)
            or any(type(v) is not int or not 0 <= v < n for v, n in zip(fields, SIZES))
            or type(row['gold']) is not int or row['gold'] not in (0, 1)):
        raise ValueError('Invalid supervision in matched population')
    return tuple(fields), row['gold']


def matched_population(base, additions, language, *, seed=1073):
    """Return identical base slots plus label-matched control/treatment slots.

    Rows keep their source identities. Repeated control inputs are explicitly
    represented by separate occurrence slots; they are not new examples.
    Development rows and their order are identical in both arms.
    """
    if language not in ('ro', 'uk') or type(seed) is not int or seed < 0:
        raise ValueError('Invalid language or matching seed')
    if not base or not additions:
        raise ValueError('Empty matched population')
    ids, tokens = set(), set()
    pools = defaultdict(list)
    for source, rows in (('base', base), ('new', additions)):
        for index, row in enumerate(rows):
            if (row['language'] != language or row['partition'] not in ('train', 'dev')
                    or source == 'new' and (row['partition'] != 'train'
                        or row.get('synthetic_only') is not True
                        or row.get('independently_annotated') is not False)):
                raise ValueError('Matched population language, partition or provenance differs')
            key = label_key(row)
            if (not isinstance(row['case_id'], str) or not row['case_id']
                    or row['case_id'] in ids or row['token_sha256'] in tokens):
                raise ValueError('Repeated source identity or input, including development overlap')
            ids.add(row['case_id']); tokens.add(row['token_sha256'])
            if (not isinstance(row['tokens'], list) or not row['tokens']
                    or any(type(v) is not int or v < 0 for v in row['tokens'])
                    or type(row['token_count']) is not int or row['token_count'] != len(row['tokens'])
                    or row['token_sha256'] != digest(row['tokens'])):
                raise ValueError('Source token identity differs')
            if source == 'base' and row['partition'] == 'train':
                pools[key].append(index)
    if {r['partition'] for r in base} != {'train', 'dev'}:
        raise ValueError('Base population must retain both TRAIN and DEV')
    # Hash ordering is stable across Python/numpy versions and does not consume
    # the optimizer's random stream. Cycling balances reuse within each cell.
    for key, indices in pools.items():
        indices.sort(key=lambda i: (digest([seed, language, key, base[i]['case_id']]), base[i]['case_id']))
    control = [('base', i) for i in range(len(base))]
    treatment = list(control)
    used = Counter(); assignments = []
    for index, row in enumerate(additions):
        key = label_key(row)
        if not pools[key]:
            raise ValueError('New label combination has no original TRAIN control')
        matched = pools[key][used[key] % len(pools[key])]
        used[key] += 1
        control.append(('base', matched)); treatment.append(('new', index))
        assignments.append(dict(slot=len(base)+index, treatment_case_id=row['case_id'],
            control_case_id=base[matched]['case_id'], control_base_index=matched,
            fields=row['fields'][:], gold=row['gold']))
    def resolve(ref):
        source, index = ref
        return (base if source == 'base' else additions)[index]
    control_rows, treatment_rows = ([resolve(ref) for ref in refs] for refs in (control, treatment))
    if any(a['partition'] != b['partition'] or label_key(a) != label_key(b)
           for a, b in zip(control_rows, treatment_rows)):
        raise ValueError('Paired slot supervision differs')
    selected = Counter(entry['control_case_id'] for entry in assignments)
    cells = [dict(fields=list(key[0]), gold=key[1], additions=count,
                  control_pool=len(pools[key]), distinct_controls=min(count, len(pools[key])))
             for key, count in sorted(used.items())]
    summary = dict(language=language, seed=seed,
        base_train_rows=sum(r['partition'] == 'train' for r in base),
        development_rows=sum(r['partition'] == 'dev' for r in base),
        added_train_slots=len(additions),
        train_rows_per_arm=sum(r['partition'] == 'train' for r in control_rows),
        control_distinct_repeated_inputs=len(selected),
        control_max_additional_repetitions=max(selected.values()),
        label_cells=cells, assignment_sha256=digest(assignments),
        source_references_sha256={arm: digest(refs) for arm, refs in
                                  [('control', control), ('treatment', treatment)]},
        development_case_ids_sha256=digest([r['case_id'] for r in base if r['partition'] == 'dev']),
        training_label_sequence_sha256=digest([label_key(r) for r in control_rows if r['partition'] == 'train']))
    return dict(control=control_rows, treatment=treatment_rows,
                references=dict(control=control, treatment=treatment),
                assignments=assignments, summary=summary)
