import functools
import itertools
from collections.abc import Iterable

import numpy as np

from .btree import balanced_btree, btree_heights, btree_sides


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


def pairs_and_singles(variables, heights):
    pair_heights = []
    for variable in variables:
        if isinstance(variable, Iterable):
            pair_heights.append(heights[variable])
    if pair_heights:
        max_pair_height = max(pair_heights)

    pairs = []
    singles = []
    for variable in variables:
        if isinstance(variable, Iterable) and heights[variable] == max_pair_height:
            pairs.append(variable)
        else:
            singles.append(variable)

    return pairs, singles


def input_cumulant(singles, covariance):
    if len(singles) == 2:
        return covariance[singles[0], singles[1]]
    else:
        return 0


def sum_over_partitions(pairs, singles, cumulant_fn):
    result = 0
    for partition in connected_partitions(pairs, singles):
        term = 1
        for part in sorted(partition, key=len):
            factor = cumulant_fn(tuple(sorted(part, key=hash)))
            term *= factor
            if factor == 0:
                break
        result += term
    return result


def preprocess_covariance(covariance, *, randomize):
    assert covariance.ndim == 2
    assert covariance.shape[0] == covariance.shape[1]

    if randomize:
        perm = np.random.permutation(covariance.shape[0])
        covariance = covariance[perm, :][:, perm]

    return covariance


def cumulant_propagation(covariance, order=None, btree=None, randomize=False):
    covariance = preprocess_covariance(covariance, randomize=randomize)
    if btree is None:
        btree = balanced_btree(range(covariance.shape[0]))
    heights = btree_heights(btree)

    @functools.cache
    def cumulant_fn(variables):
        if order is not None and len(variables) > order:
            return 0
        pairs, singles = pairs_and_singles(variables, heights)
        if len(pairs) == 0:
            return input_cumulant(singles, covariance)
        return sum_over_partitions(pairs, singles, cumulant_fn)

    return cumulant_fn((btree,))


def impute(variables, max_height, sides, cumulant_fn, inv_cov_fn):
    left_vars = [var for var in variables if sides[var] == 0]
    right_vars = [var for var in variables if sides[var] == 1]
    if len(left_vars) <= 1 or len(right_vars) <= 1:
        return 0
    inv_cov, (left_cov_vars, right_cov_vars) = inv_cov_fn(max_height)
    left_cumulants = np.array(
        [cumulant_fn(tuple(left_vars) + (var,)) for var in right_cov_vars]
    )
    right_cumulants = np.array(
        [cumulant_fn((var,) + tuple(right_vars)) for var in left_cov_vars]
    )
    return (left_cumulants[None] @ inv_cov @ right_cumulants[:, None])[0, 0]


def cumulant_propagation_with_imputation(
    covariance, order=None, btree=None, randomize=False
):
    covariance = preprocess_covariance(covariance, randomize=randomize)
    if btree is None:
        btree = balanced_btree(range(covariance.shape[0]))
    heights = btree_heights(btree)
    sides = btree_sides(btree)

    @functools.cache
    def inv_cov_fn(max_height):
        left_vars = [
            var
            for var, ht in heights.items()
            if (ht <= max_height or not isinstance(var, Iterable)) and sides[var] == 0
        ]
        right_vars = [
            var
            for var, ht in heights.items()
            if (ht <= max_height or not isinstance(var, Iterable)) and sides[var] == 1
        ]
        cov = np.array(
            [
                [cumulant_fn((left_var, right_var)) for right_var in right_vars]
                for left_var in left_vars
            ]
        )
        return np.linalg.pinv(cov), (left_vars, right_vars)

    @functools.cache
    def cumulant_fn(variables):
        pairs, singles = pairs_and_singles(variables, heights)
        if len(pairs) == 0:
            return input_cumulant(singles, covariance)
        if order is not None and len(variables) > order:
            max_height = max([heights[var] for var in variables])
            return impute(variables, max_height, sides, cumulant_fn, inv_cov_fn)
        return sum_over_partitions(
            pairs, singles, lambda variables: cumulant_fn(variables)
        )

    return cumulant_fn((btree,))
