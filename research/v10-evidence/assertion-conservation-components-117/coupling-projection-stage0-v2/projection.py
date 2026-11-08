"""Stage0 assertion-only KL projection under a fixed coupled-policy score.

This operates on probability vectors, not models or labels. In exact arithmetic,
min KL(p || target) subject to p>=0, sum(p)=1 and g.p=0 preserves the parent's
coupled-policy score. Numerical outputs are checked against the direct score;
identity fallback preserves the parent if support, solving or rounding fails.
"""
import math

FIELD_SIZES = dict(subject=3, time=3, assertion=4, source=3, relation=2, concern=2)
A, N, U, R = range(4)


def _probabilities(values, size, name):
    if (not isinstance(values, (tuple, list)) or len(values) != size
            or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or abs(math.fsum(values) - 1.) > 1e-12):
        raise ValueError('Invalid probability vector: ' + name)
    return list(values)


def validate(fields, couplings):
    if not isinstance(fields, dict) or set(fields) != set(FIELD_SIZES):
        raise ValueError('Exactly six field vectors required')
    copied = {name: _probabilities(fields[name], size, name) for name, size in FIELD_SIZES.items()}
    if not isinstance(couplings, dict) or set(couplings) != {'scope', 'assertion_relation'}:
        raise ValueError('Unexpected coupling inventory')
    for name, shape in (('scope', (3, 3)), ('assertion_relation', (4, 2))):
        table = couplings[name]
        if (not isinstance(table, list) or len(table) != shape[0]
                or any(not isinstance(row, list) or len(row) != shape[1] for row in table)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for row in table for v in row)):
            raise ValueError('Invalid positive coupling table: ' + name)
    return copied


def _pair(left, right, table):
    matrix = [[x * y * table[i][j] for j, y in enumerate(right)] for i, x in enumerate(left)]
    total = sum(sum(row) for row in matrix)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Invalid coupled normalization')
    return [[v / total for v in row] for row in matrix]


def score(fields, couplings):
    """Direct serving formula, with the frozen portable scorer's operation order.

    Keeping this order matters at exact threshold boundaries. Tests also compare
    against the repository scorer; coefficient derivation is a separate path.
    """
    scope = _pair(fields['subject'], fields['source'], couplings['scope'])
    ar = _pair(fields['assertion'], fields['relation'], couplings['assertion_relation'])
    mismatch = sum(scope[i][j] for i in range(3) for j in range(3) if (i == 2) != (j == 2))
    real = sum(scope[i][j] for i in range(3) for j in range(3) if i != 2 and j != 2)
    fiction_review = 1 - sum(ar[i][0] for i in range(4) if i != R)
    real_review = 1 - ar[N][0]
    return max(0., min(1., fields['concern'][1] * (mismatch + scope[2][2] * fiction_review + real * real_review)))


def coefficients(fields, couplings, parent_score):
    """S(q)=concern*(a.q)/(b.q); conservation is (concern*a-S0*b).q=0."""
    scope = _pair(fields['subject'], fields['source'], couplings['scope'])
    mismatch = math.fsum(scope[i][j] for i in range(3) for j in range(3) if (i == 2) != (j == 2))
    real = math.fsum(scope[i][j] for i in range(2) for j in range(2))
    fiction = scope[2][2]
    r = fields['relation']
    table = couplings['assertion_relation']
    a, b = [], []
    for i in range(4):
        b_i = math.fsum(r[j] * table[i][j] for j in range(2))
        fiction_i = b_i if i == R else r[1] * table[i][1]
        real_i = r[1] * table[i][1] if i == N else b_i
        b.append(b_i)
        a.append(mismatch * b_i + fiction * fiction_i + real * real_i)
    g = [fields['concern'][1] * x - parent_score * y for x, y in zip(a, b)]
    return a, b, g


def _tilt(target, scaled, multiplier):
    active = [i for i, value in enumerate(target) if value > 0]
    logs = [math.log(target[i]) - multiplier * scaled[i] for i in active]
    peak = max(logs)
    weights = [math.exp(value - peak) for value in logs]
    total = math.fsum(weights)
    p = [0.] * 4
    for i, value in zip(active, weights):
        p[i] = value / total
    return p, math.fsum(x * y for x, y in zip(p, scaled))


def _kl(p, target):
    if any(x > 0 and y == 0 for x, y in zip(p, target)):
        return None  # Parent fallback may lie outside the requested KL support.
    return math.fsum(x * (math.log(x) - math.log(y)) for x, y in zip(p, target) if x > 0)


def project_assertion(fields, target, couplings, *, threshold, risk_tolerance=1e-12,
                      maximum_iterations=256, maximum_bracket_steps=64):
    """Return an explicitly diagnosed projection or the exact parent vectors.

    Zero target components are structural support restrictions, never smoothed.
    There is no label-dependent gate, row exclusion, threshold search or fitting.
    Only assertion probabilities can change; all five other vectors are copied.
    """
    original = validate(fields, couplings)
    target = _probabilities(target, 4, 'target_assertion')
    if (type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 <= threshold <= 1
            or type(risk_tolerance) not in (float, int) or not math.isfinite(risk_tolerance) or not 0 < risk_tolerance <= 1e-10
            or type(maximum_iterations) is not int or maximum_iterations < 1
            or type(maximum_bracket_steps) is not int or maximum_bracket_steps < 1):
        raise ValueError('Invalid fixed numerical contract')
    parent = list(original['assertion'])
    initial_score = score(original, couplings)
    a, b, g = coefficients(original, couplings, initial_score)
    scale = max(abs(x) for x in g)
    iterations = bracket_steps = 0
    multiplier = None

    def finish(assertion, status, reason=None):
        out = {key: list(value) for key, value in original.items()}
        out['assertion'] = list(assertion)
        actual = score(out, couplings)
        difference = actual - initial_score
        same_sign = (actual >= threshold) == (initial_score >= threshold)
        return dict(fields=out, status=status, fallback_reason=reason,
            parent_score=initial_score, projected_score=actual, score_difference=difference,
            binary_decision_unchanged=same_sign, immutable_other_fields=True,
            iterations=iterations, bracket_steps=bracket_steps, scaled_multiplier=multiplier,
            affine_residual=math.fsum(x * y for x, y in zip(g, assertion)),
            parent_affine_residual=math.fsum(x * y for x, y in zip(g, parent)),
            kl_to_target=_kl(assertion, target), mathematical_constraint='simplex and (concern*a-S0*b).p=0',
            numerical_conservation_tolerance=risk_tolerance, qualified=False)

    def accept(p, status):
        if (any(not math.isfinite(x) or x < 0 or x > 1 for x in p)
                or abs(math.fsum(p) - 1.) > 1e-12):
            return finish(parent, 'IDENTITY_FALLBACK', 'invalid_projection_simplex')
        result = finish(p, status)
        if abs(result['score_difference']) > risk_tolerance:
            return finish(parent, 'IDENTITY_FALLBACK', 'score_tolerance')
        if not result['binary_decision_unchanged']:
            return finish(parent, 'IDENTITY_FALLBACK', 'threshold_rounding')
        parent_kl = _kl(parent, target)
        if parent_kl is not None and result['kl_to_target'] > parent_kl + 1e-10:
            return finish(parent, 'IDENTITY_FALLBACK', 'objective_not_improved')
        return result

    if target == parent:
        return finish(parent, 'IDENTITY_TARGET')
    if scale == 0:
        return accept(target, 'NUMERICALLY_FLAT_CONSTRAINT')
    scaled = [value / scale for value in g]
    if min(scaled) > 0 or max(scaled) < 0:
        # The parent is feasible in exact arithmetic. Same-sign coefficients
        # across the entire simplex therefore reflect numerical cancellation,
        # not a limitation of the requested target support.
        return finish(parent, 'IDENTITY_FALLBACK', 'numerical_parent_constraint_inconsistency')
    support = [i for i, value in enumerate(target) if value > 0]
    lower, upper = min(scaled[i] for i in support), max(scaled[i] for i in support)
    if lower > 0 or upper < 0:
        return finish(parent, 'IDENTITY_FALLBACK', 'target_support_excludes_affine_constraint')
    if lower == 0 or upper == 0:
        face = [i for i in support if scaled[i] == 0]
        if not face:
            return finish(parent, 'IDENTITY_FALLBACK', 'numerical_degenerate_face')
        mass = math.fsum(target[i] for i in face)
        p = [target[i] / mass if i in face else 0. for i in range(4)]
        return accept(p, 'BOUNDARY_FACE_PROJECTION')

    # For positive target mass on both signs, E_p[g] is monotone decreasing in
    # lambda, derivative -Var_p(g); the finite root is unique unless g is constant.
    left, right = -1., 1.
    p_left, mean_left = _tilt(target, scaled, left)
    p_right, mean_right = _tilt(target, scaled, right)
    while (mean_left < 0 or mean_right > 0) and bracket_steps < maximum_bracket_steps:
        if mean_left < 0:
            left *= 2
            p_left, mean_left = _tilt(target, scaled, left)
        if mean_right > 0:
            right *= 2
            p_right, mean_right = _tilt(target, scaled, right)
        bracket_steps += 1
    if mean_left < 0 or mean_right > 0:
        return finish(parent, 'IDENTITY_FALLBACK', 'root_not_bracketed')
    best, best_mean = (p_left, mean_left) if abs(mean_left) <= abs(mean_right) else (p_right, mean_right)
    best_multiplier = left if abs(mean_left) <= abs(mean_right) else right
    solved = False
    for iterations in range(1, maximum_iterations + 1):
        multiplier = (left + right) / 2
        p, mean = _tilt(target, scaled, multiplier)
        if abs(mean) < abs(best_mean):
            best, best_mean = p, mean
            best_multiplier = multiplier
        if mean == 0 or right - left <= 2e-15 * max(1., abs(left), abs(right)):
            solved = True
            break
        if mean > 0:
            left = multiplier
        else:
            right = multiplier
    if not solved:
        return finish(parent, 'IDENTITY_FALLBACK', 'iteration_limit')
    multiplier = best_multiplier
    return accept(best, 'KL_AFFINE_PROJECTION')


def conditional_au_target(parent_assertion, residual):
    """Zero-residual identity; only the proposed target's A/U split changes.

    The subsequent fixed-score projection may also move N/R. This helper is not
    itself score-preserving, and it does not load, train or invoke a model.
    """
    parent = _probabilities(parent_assertion, 4, 'parent_assertion')
    if type(residual) not in (int, float) or not math.isfinite(residual):
        raise ValueError('Residual must be finite')
    if residual == 0 or parent[A] + parent[U] == 0:
        return parent
    if parent[A] == 0 or parent[U] == 0:
        return parent  # Finite log-odds correction cannot introduce absent support.
    logit = math.log(parent[U]) - math.log(parent[A]) + residual
    ratio = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
    mass = parent[A] + parent[U]
    output = list(parent)
    output[U] = mass * ratio
    output[A] = mass - output[U]
    return output
