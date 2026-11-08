"""Independent finite-vertex oracle for synthetic fixed-marginal transport checks.

Each positive-concern mass x_i lies in [0, q_i], and sum(x_i)=p.
A vertex has at most one coordinate strictly between its two bounds.
Use Decimal arithmetic on the supplied decimal representations; do not load
model outputs, metadata, features or training/evaluation examples here.
"""
from decimal import Decimal, localcontext


def vertex_bounds(capacities, positive_mass, event_weights):
    if len(capacities) != 9 or len(event_weights) != 9:
        raise ValueError('Expected nine scope cells')
    with localcontext() as context:
        context.prec = 70
        q = [Decimal(str(x)) for x in capacities]
        w = [Decimal(str(x)) for x in event_weights]
        p = Decimal(str(positive_mass))
        if any(not x.is_finite() or x < 0 for x in q):
            raise ValueError('Invalid capacity')
        if any(not x.is_finite() or x < 0 or x > 1 for x in w):
            raise ValueError('Invalid event weight')
        if not p.is_finite() or not 0 <= p <= sum(q):
            raise ValueError('Infeasible positive mass')
        low = high = None
        low_vertex = high_vertex = None
        feasible_vertices = 0
        for free in range(9):
            others = [i for i in range(9) if i != free]
            for mask in range(1 << 8):
                vector = [Decimal(0)] * 9
                for j, index in enumerate(others):
                    if mask & (1 << j):
                        vector[index] = q[index]
                remainder = p - sum(vector)
                if remainder < 0 or remainder > q[free]:
                    continue
                vector[free] = remainder
                objective = sum(x*y for x, y in zip(vector, w))
                feasible_vertices += 1
                if low is None or objective < low:
                    low, low_vertex = objective, vector
                if high is None or objective > high:
                    high, high_vertex = objective, vector
        if low is None:
            raise AssertionError('Feasible bounded polytope must contain a vertex')
        return {
            'minimum': float(low), 'maximum': float(high),
            'minimum_decimal': str(low), 'maximum_decimal': str(high),
            'minimum_positive_cells': [float(x) for x in low_vertex],
            'maximum_positive_cells': [float(x) for x in high_vertex],
            'enumerated_feasible_vertices': feasible_vertices,
            'real_data_used': False,
        }


if __name__ == '__main__':
    # Dyadic capacities/mass have exact decimal sums as well as exact binary
    # representation, separating oracle logic from simplex-rounding ambiguity.
    result = vertex_bounds([.125]*8+[0.], .5, [0.,1.,0.,1.,0.,1.,0.,1.,.5])
    assert result['minimum'] == 0. and result['maximum'] == .5
    assert vertex_bounds([1.]+[0.]*8, 1., [.75]*9)['minimum'] == .75
    assert vertex_bounds([1.]+[0.]*8, 0., [.75]*9)['maximum'] == 0.
    print('Three synthetic oracle sanity checks passed; no model/data loaded.')
