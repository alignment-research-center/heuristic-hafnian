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
    max_height = max(heights[variable] for variable in variables)

    pairs = []
    singles = []
    for variable in variables:
        if isinstance(variable, Iterable) and heights[variable] == max_height:
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


def extended_covariance_propagation_v1(
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
    left_vars = []
    right_vars = []
    if 1 in exponents:
        left_vars += list(range(n))
        right_vars += list(range(n, 2 * n))
    if 2 in exponents:
        left_vars += [(i, j) for i, j in itertools.product(*(range(n),) * 2) if i < j]
        right_vars += [
            (i, j) for i, j in itertools.product(*(range(n, 2 * n),) * 2) if i < j
        ]
    # This ensures linearity but is a bit of a hack
    for var in left_vars + right_vars:
        heights[var] = 1 if isinstance(var, Iterable) else 0
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


def extended_covariance_propagation_v2(covariance, btree=None, randomize=False):
    covariance = preprocess_covariance(covariance, randomize=randomize)
    assert covariance.shape[0] % 2 == 0
    n = covariance.shape[0] // 2
    if btree is None:
        btree = balanced_btree(range(covariance.shape[0]))

    levels = {}
    heights = {}
    for height, level in enumerate(reversed(list(btree_levels(btree)))):
        levels[height] = level
        for var in level:
            heights[var] = height

    @functools.cache
    def cumulant(*variables):
        var_heights = [heights[var] for var in variables]
        max_height = max(var_heights)
        pairs, singles = [], []
        for variable, height in zip(variables, var_heights):
            (
                pairs
                if isinstance(variable, Iterable) and height == max_height
                else singles
            ).append(variable)

        if len(pairs) == 0:
            if len(singles) == 2:
                return covariance[singles[0], singles[1]]
            else:
                return 0

        if len(pairs) == 2 and len(singles) == 0:
            [level_height] = [ht for ht, lvl in levels.items() if pairs[0] in lvl]
            left_pairs = [p for p in pairs if all(i < n for i in flatten_btree(p))]
            right_pairs = [p for p in pairs if all(i >= n for i in flatten_btree(p))]
            if len(left_pairs) == 1 and len(right_pairs) == 1:
                [left_pair] = left_pairs
                [right_pair] = right_pairs
                left_vars = [
                    var
                    for ht, lvl in levels.items()
                    for var in lvl
                    if (
                        ht < level_height
                        or ht == level_height
                        and not isinstance(var, Iterable)
                    )
                    and all(i < n for i in flatten_btree(var))
                ]
                right_vars = [
                    var
                    for ht, lvl in levels.items()
                    for var in lvl
                    if (
                        ht < level_height
                        or ht == level_height
                        and not isinstance(var, Iterable)
                    )
                    and all(i >= n for i in flatten_btree(var))
                ]
                # Using these variables instead should make the algorithm the same as v1
                # left_vars = list(range(n))
                # right_vars = list(range(n, 2 * n))
                left_cov = np.array(
                    [cumulant(left_pair, right_var) for right_var in right_vars]
                )[None]
                right_cov = np.array(
                    [cumulant(right_pair, left_var) for left_var in left_vars]
                )[:, None]
                mid_cov = np.array(
                    [
                        [cumulant(left_var, right_var) for right_var in right_vars]
                        for left_var in left_vars
                    ]
                )
                return (left_cov @ np.linalg.pinv(mid_cov) @ right_cov)[0, 0]

        result = 0
        for partition in connected_partitions(pairs, singles):
            if any(len(part) > 3 for part in partition):
                continue
            result += np.prod([cumulant(*sorted(part, key=hash)) for part in partition])
        return result

    return cumulant(btree)
