"""Research sampler with two TRAIN examples per origin/language in each update.

Every generated example is visited once per pass. Historical strata are
reshuffled and repeated to the same length. Category weights balance the five
labels in expectation within each stratum; they do not balance model gradients
or guarantee better accuracy. This module does not alter existing trainers.
"""
from collections import Counter
from numbers import Integral

from roguard.trajectory_model import RESPONSE_LABELS


STRATA = tuple((origin, language) for origin in ('historical', 'generated')
               for language in ('ro', 'uk'))


class MixedResponseSampler:
    """Validate fixed TRAIN identities before constructing any shuffled batch.

    ``rows`` may include DEV records, but ``indices`` must contain only unique
    TRAIN records. Returned indices address the original rows and feature arrays.
    The equal, even generated-stratum sizes and even historical sizes ensure
    complete generated coverage and no duplicate example within an update.
    """

    def __init__(self, rows, indices):
        import numpy as np

        indices = list(indices)
        if (not indices or any(isinstance(i, bool) or not isinstance(i, Integral)
                               or not 0 <= i < len(rows) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Expected unique valid TRAIN indices')
        groups = Counter()
        strata = {key: [] for key in STRATA}
        identities = set()
        for i in indices:
            row = rows[i]
            key = (row.get('origin'), row.get('language'))
            case_id = row.get('case_id')
            if (row.get('partition') != 'train' or key not in strata
                    or row.get('gold') not in RESPONSE_LABELS
                    or not isinstance(case_id, str) or not case_id
                    or case_id in identities):
                raise ValueError('Invalid, duplicated, or non-TRAIN sampler record')
            identities.add(case_id)
            strata[key].append(i)
            groups[key + (row['gold'],)] += 1
        if len(groups) != 20:
            raise ValueError('Expected all 20 TRAIN origin/language/category groups')
        target = len(strata[('generated', 'ro')])
        if (target != len(strata[('generated', 'uk')])
                or any(len(s) % 2 or not 2 <= len(s) <= target for s in strata.values())):
            raise ValueError('Expected equal generated sizes and even, nonempty strata no larger than generated')
        self._strata = tuple(np.asarray(strata[key], dtype=np.int64) for key in STRATA)
        for s in self._strata:
            s.setflags(write=False)
        weights = np.zeros(len(rows), dtype=np.float32)
        for i in indices:
            row = rows[i]
            key = (row['origin'], row['language'])
            weights[i] = len(strata[key]) / (5 * groups[key + (row['gold'],)])
        weights.setflags(write=False)
        self.weights = weights
        self.batches_per_pass = target // 2
        self.presentations_per_pass = 4 * target
        self._target = target

    def batches(self, rng):
        """Draw one complete pass using the caller's persistent NumPy generator."""
        import numpy as np

        columns = []
        for indices in self._strata:
            chunks = []
            remaining = self._target
            while remaining:
                draw = rng.permutation(indices)[:remaining]
                chunks.append(draw)
                remaining -= len(draw)
            columns.append(np.concatenate(chunks).reshape(-1, 2))
        return np.concatenate(columns, axis=1)
