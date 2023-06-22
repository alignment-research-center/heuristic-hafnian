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


def unbalanced_btree(items):
    if len(items) == 1:
        return items[0]
    return (unbalanced_btree(items[:1]), unbalanced_btree(items[1:]))


def flatten_btree(btree):
    if isinstance(btree, Iterable):
        return tuple(item for subtree in btree for item in flatten_btree(subtree))
    else:
        return (btree,)


def btree_levels(btree):
    btree = (btree,)
    yield btree
    while any(isinstance(subtree, Iterable) for subtree in btree):
        btree = tuple(
            item
            for subtree in btree
            if isinstance(subtree, Iterable)
            for item in subtree
        )
        yield btree


def btree_heights(btree):
    heights = {}
    for height, level in enumerate(reversed(list(btree_levels(btree)))):
        for subtree in level:
            heights[subtree] = height
    return heights


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


def covariance_propagation(covariance, btree=None, randomize=False):
    return cumulant_propagation(covariance, order=2, btree=btree, randomize=randomize)


def extended_covariance_propagation(
    covariance,
    btree=None,
    randomize=False,
    exponents=(1,),
    order=3,
    order_mid=None,
):
    covariance = preprocess_covariance(covariance, randomize=randomize)
    assert covariance.shape[0] % 2 == 0
    n = covariance.shape[0] // 2
    if btree is None:
        btree = balanced_btree(range(covariance.shape[0]))
    heights = btree_heights(btree)

    @functools.cache
    def cumulant_fn(variables, order):
        if order is not None and len(variables) > order:
            return 0
        pairs, singles = pairs_and_singles(variables, heights)
        if len(pairs) == 0:
            return input_cumulant(singles, covariance)
        return sum_over_partitions(
            pairs, singles, lambda variables: cumulant_fn(variables, order)
        )

    left_ev = cumulant_fn((btree[0],), order=order)
    right_ev = cumulant_fn((btree[1],), order=order)
    left_vars = {
        1: list(range(n)),
        2: [(i, j) for i, j in itertools.product(*(range(n),) * 2) if i < j],
        3: [
            (i, (j, k))
            for i, j, k in itertools.product(*(range(n),) * 3)
            if i < j and j < k
        ],
    }
    right_vars = {
        1: list(range(n, 2 * n)),
        2: [(i, j) for i, j in itertools.product(*(range(n, 2 * n),) * 2) if i < j],
        3: [
            (i, (j, k))
            for i, j, k in itertools.product(*(range(n, 2 * n),) * 3)
            if i < j and j < k
        ],
    }
    # This ensures linearity but is a bit of a hack
    for exponent in left_vars.keys():
        for var in left_vars[exponent] + right_vars[exponent]:
            heights[var] = exponent - 1
    left_vars = [var for exponent in exponents for var in left_vars[exponent]]
    right_vars = [var for exponent in exponents for var in right_vars[exponent]]
    left_cov = np.array(
        [cumulant_fn((btree[0], var), order=order) for var in right_vars]
    )[None]
    right_cov = np.array(
        [cumulant_fn((var, btree[1]), order=order) for var in left_vars]
    )[:, None]
    mid_cov = np.array(
        [
            [
                cumulant_fn((left_var, right_var), order=order_mid)
                for right_var in right_vars
            ]
            for left_var in left_vars
        ]
    )
    if mid_cov.size == 0:
        cov = 0
    else:
        cov = (left_cov @ np.linalg.pinv(mid_cov) @ right_cov)[0, 0]
    return left_ev * right_ev + cov


def extended_third_cumulant_propagation(covariance, btree=None, randomize=False):
    covariance = preprocess_covariance(covariance, randomize=randomize)
    assert covariance.shape[0] % 2 == 0
    n = covariance.shape[0] // 2
    if btree is None:
        btree = balanced_btree(range(covariance.shape[0]))
    heights = btree_heights(btree)
    order = 3

    @functools.cache
    def cumulant_fn(variables):
        if order is not None and len(variables) > order:
            return 0
        pairs, singles = pairs_and_singles(variables, heights)
        if len(pairs) == 0:
            return input_cumulant(singles, covariance)
        result = sum_over_partitions(
            pairs, singles, lambda variables: cumulant_fn(variables)
        )

        if not (len(pairs) == 2 and len(singles) == 0):
            return result

        left_pairs = [p for p in pairs if all(i < n for i in flatten_btree(p))]
        right_pairs = [p for p in pairs if all(i >= n for i in flatten_btree(p))]
        if not (len(left_pairs) == 1 and len(right_pairs) == 1):
            return result
        [left_pair] = left_pairs
        [right_pair] = right_pairs

        left_vars = [
            var
            for var, ht in heights.items()
            if (
                ht < heights[left_pair]
                or (ht == heights[left_pair] and (not isinstance(var, Iterable)))
            )
            and all(i < n for i in flatten_btree(var))
        ]
        right_vars = [
            var
            for var, ht in heights.items()
            if (
                ht < heights[right_pair]
                or (ht == heights[right_pair] and (not isinstance(var, Iterable)))
            )
            and all(i >= n for i in flatten_btree(var))
        ]
        left_cov = np.array(
            [
                cumulant_fn((left_pair[0], left_pair[1], right_var))
                for right_var in right_vars
            ]
        )[None]
        right_cov = np.array(
            [
                cumulant_fn((right_pair[0], right_pair[1], left_var))
                for left_var in left_vars
            ]
        )[:, None]
        mid_cov = np.array(
            [
                [cumulant_fn((left_var, right_var)) for right_var in right_vars]
                for left_var in left_vars
            ]
        )
        result += (left_cov @ np.linalg.pinv(mid_cov) @ right_cov)[0, 0]

        return result

    return cumulant_fn((btree,))
