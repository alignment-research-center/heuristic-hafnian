import functools
import itertools
from collections.abc import Iterable

import numpy as np

from .btree import balanced_btree, breadth_first, btree_height, flatten_btree


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
    max_height = max([heights[var] for var in variables])

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


def get_heights(btrees):
    heights = {}
    for btree in btrees:
        for subtree in breadth_first(btree):
            heights[subtree] = btree_height(subtree)
    return heights


def cumulant_propagation(
    covariance, order=None, btree=None, variables=None, randomize=False
):
    covariance = preprocess_covariance(covariance, randomize=randomize)
    if btree is None:
        btree = balanced_btree(range(covariance.shape[0]))
    if variables is None:
        variables = (btree,)
    heights = get_heights((btree,) + variables)

    @functools.cache
    def cumulant_fn(variables):
        if order is not None and len(variables) > order:
            return 0
        pairs, singles = pairs_and_singles(variables, heights)
        if len(pairs) == 0:
            return input_cumulant(singles, covariance)
        return sum_over_partitions(pairs, singles, cumulant_fn)

    return cumulant_fn(variables)


def get_sides(btree, variables):
    left_inputs = flatten_btree(btree[0])
    right_inputs = flatten_btree(btree[1])
    sides = {}
    for subtree in breadth_first(btree[0]):
        sides[subtree] = 0
    for subtree in breadth_first(btree[1]):
        sides[subtree] = 1
    for variable in variables:
        for subtree in breadth_first(variable):
            inputs = flatten_btree(subtree)
            if all(i in left_inputs for i in inputs):
                sides[subtree] = 0
            elif all(i in right_inputs for i in inputs):
                sides[subtree] = 1
    return sides


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
    covariance, order=None, btree=None, variables=None, randomize=False
):
    covariance = preprocess_covariance(covariance, randomize=randomize)
    if btree is None:
        btree = balanced_btree(range(covariance.shape[0]))
    if variables is None:
        variables = (btree,)
    heights = get_heights((btree,) + variables)
    sides = get_sides(btree, variables)

    @functools.cache
    def inv_cov_fn(max_height):
        left_vars = [
            var
            for var in breadth_first(btree)
            if heights[var] <= max_height and sides[var] == 0
        ]
        right_vars = [
            var
            for var in breadth_first(btree)
            if heights[var] <= max_height and sides[var] == 1
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
        return sum_over_partitions(pairs, singles, cumulant_fn)

    return cumulant_fn(variables)


def permanent(covariance, btree=None):
    import itertools

    if btree is None:
        btree = balanced_btree(range(covariance.shape[0]))

    @functools.cache
    def cov_fn(var1, var2):
        height1 = btree_height(var1)
        height2 = btree_height(var2)
        assert height1 == height2

        if height1 == 0:
            return covariance[var1, var2]

        if height1 == 1:
            return (
                covariance[var1[0], var2[0]] * covariance[var1[1], var2[1]]
                + covariance[var1[0], var2[1]] * covariance[var1[1], var2[0]]
            )

        if height1 > 1:
            result = 0
            subvar1 = var1[0] + var1[1]
            subvar2 = var2[0] + var2[1]
            for perm in itertools.permutations(range(4)):
                result -= (
                    2
                    * cov_fn(subvar1[0], subvar2[perm[0]])
                    * cov_fn(subvar1[1], subvar2[perm[1]])
                    * cov_fn(subvar1[2], subvar2[perm[2]])
                    * cov_fn(subvar1[3], subvar2[perm[3]])
                )
                result += 0.25 * (
                    (
                        cov_fn(
                            (subvar1[0], subvar1[1]),
                            (subvar2[perm[0]], subvar2[perm[1]]),
                        )
                        * cov_fn(
                            (subvar1[2], subvar1[3]),
                            (subvar2[perm[2]], subvar2[perm[3]]),
                        )
                    )
                    + (
                        cov_fn(
                            (subvar1[0], subvar1[2]),
                            (subvar2[perm[0]], subvar2[perm[2]]),
                        )
                        * cov_fn(
                            (subvar1[1], subvar1[3]),
                            (subvar2[perm[1]], subvar2[perm[3]]),
                        )
                    )
                    + (
                        cov_fn(
                            (subvar1[0], subvar1[3]),
                            (subvar2[perm[0]], subvar2[perm[3]]),
                        )
                        * cov_fn(
                            (subvar1[2], subvar1[1]),
                            (subvar2[perm[2]], subvar2[perm[1]]),
                        )
                    )
                )
            return result

    return cov_fn(btree, btree)
