import functools
import itertools
from collections.abc import Iterable

import numpy as np


def connected_partitions(pairs, singles=None):
    """
    Given a list of pairs and a list of singles, yields all
    partitions of the union of the pairs union the singles
    that are connected after including the pairs.
    """
    pairs = list(pairs)
    if singles is None:
        singles = []
    singles = list(singles)

    if len(pairs) == 0:
        if len(singles) == 0:
            yield []
        else:
            yield [singles]
        return

    # First case: partition is connected even without the last pair.
    last_pair = pairs.pop()
    yield from connected_partitions(pairs, singles + list(last_pair))

    def sub_partitions(i, pairs_indices, singles_indices):
        return connected_partitions(
            [item for item, index in zip(pairs, pairs_indices) if index == i],
            [item for item, index in zip(singles, singles_indices) if index == i]
            + [last_pair[i]],
        )

    # Second case: partition requires the last pair to be connected.
    # In this case, the partition splits into two connected components without this pair.
    for pairs_indices in itertools.product(*(range(2),) * len(pairs)):
        for singles_indices in itertools.product(*(range(2),) * len(singles)):
            for partition_0 in sub_partitions(0, pairs_indices, singles_indices):
                for partition_1 in sub_partitions(1, pairs_indices, singles_indices):
                    yield partition_0 + partition_1


def balanced_btree(items):
    if len(items) == 1:
        return items[0]
    mid = (len(items) + 1) // 2
    return (balanced_btree(items[:mid]), balanced_btree(items[mid:]))


def cumulant_propagation(covariance, order=None, btree=None):
    assert covariance.ndim == 2
    assert covariance.shape[0] == covariance.shape[1]

    if btree is None:
        indices = np.arange(covariance.shape[0])
        np.random.shuffle(indices)
        btree = balanced_btree(indices)

    @functools.cache
    def cumulant(*variables):
        pairs, singles = [], []
        for variable in variables:
            (pairs if isinstance(variable, Iterable) else singles).append(variable)

        if len(pairs) == 0:
            if len(singles) == 2:
                return covariance[singles[0], singles[1]]
            else:
                return 0

        result = 0
        for partition in connected_partitions(pairs, singles):
            if order is not None and any(len(part) > order for part in partition):
                continue
            result += np.prod([cumulant(*sorted(part, key=hash)) for part in partition])
        return result

    return cumulant(btree)
