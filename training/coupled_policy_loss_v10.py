"""MLX supervision of the existing coupled-policy probability, without an encoder.

The policy is unchanged from v9. This helper makes its served probability
differentiable for the separate large-data readout experiment.
"""
from roguard.evidence import ONTOLOGY
from training.native_fields_v9 import FIELDS, SIZES

LOSS_EPSILON = 1e-6


def make_coupled_policy_mass(mx, couplings):
    """Validate constant pair potentials once and return a differentiable scorer."""
    import numpy as np

    expected = {
        'scope': ('subject', 'source'),
        'assertion_relation': ('assertion', 'relation'),
    }
    if not isinstance(couplings, dict) or set(couplings) != set(expected):
        raise ValueError('Expected exactly the two registered coupling tables')
    weights = {}
    for name, (left, right) in expected.items():
        value = np.asarray(couplings[name], dtype=np.float32)
        shape = (len(ONTOLOGY[left]), len(ONTOLOGY[right]))
        if value.shape != shape or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError('Invalid coupling table: ' + name)
        weights[name] = mx.array(value)
    boundaries = list(np.cumsum(SIZES)[:-1])
    sf = ONTOLOGY['subject'].index('FICTIONAL')
    fi = ONTOLOGY['source'].index('FICTION')
    neg = ONTOLOGY['assertion'].index('NEGATED')
    ret = ONTOLOGY['assertion'].index('RETRACTED')
    new = ONTOLOGY['relation'].index('NEW')

    def mass(logits):
        if logits.ndim != 2 or logits.shape[1] != sum(SIZES) or logits.shape[0] == 0:
            raise ValueError('Expected nonempty six-field semantic logits')
        values = [mx.softmax(part, axis=-1) for part in mx.split(logits, boundaries, axis=-1)]
        pairs = {}
        for name, (left, right) in expected.items():
            joint = (values[FIELDS.index(left)][:, :, None]
                     * values[FIELDS.index(right)][:, None, :] * weights[name][None, :, :])
            pairs[name] = joint / mx.sum(joint, axis=(1, 2), keepdims=True)
        scope, ar = pairs['scope'], pairs['assertion_relation']
        mismatch = sum(scope[:, i, j] for i in range(scope.shape[1])
                       for j in range(scope.shape[2]) if (i == sf) != (j == fi))
        real = sum(scope[:, i, j] for i in range(scope.shape[1])
                   for j in range(scope.shape[2]) if i != sf and j != fi)
        fiction_review = 1 - sum(ar[:, i, new] for i in range(ar.shape[1]) if i != ret)
        concern = values[FIELDS.index('concern')][:, ONTOLOGY['concern'].index('1')]
        return mx.clip(concern * (mismatch + scope[:, sf, fi] * fiction_review
                                 + real * (1 - ar[:, neg, new])), 0, 1)

    return mass


def policy_binary_cross_entropy(mx, probabilities, labels):
    """Unreduced BCE with a fixed, explicit probability clamp."""
    if probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError('Policy probability and label shapes differ')
    bounded = mx.clip(probabilities, LOSS_EPSILON, 1 - LOSS_EPSILON)
    return -(labels * mx.log(bounded) + (1 - labels) * mx.log1p(-bounded))
