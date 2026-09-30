"""Pure class-balancing weights for applicable V27 latent facts."""

from __future__ import annotations

from roguard.v26_facts import truth_vector


def fact_value_weights(rows: list[dict]) -> tuple[list[float], list[float]]:
    positive = [0] * 6
    negative = [0] * 6
    for row in rows:
        truth, mask = truth_vector(row)
        for index in range(6):
            if mask[index]:
                if truth[index]:
                    positive[index] += 1
                else:
                    negative[index] += 1
    if any(not pos or not neg for pos, neg in zip(positive, negative)):
        raise ValueError("V27 requires both truth values for every applicable fact")
    total = [pos + neg for pos, neg in zip(positive, negative)]
    return ([count / (2 * pos) for count, pos in zip(total, positive)],
            [count / (2 * neg) for count, neg in zip(total, negative)])
