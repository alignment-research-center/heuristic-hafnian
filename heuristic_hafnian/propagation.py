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


import numpy as np
import itertools


def symmetrize(a):
    assert a.ndim % 2 == 0
    b = 0
    for perm in itertools.permutations(range(a.ndim // 2)):
        b += a.transpose(perm + tuple(range(a.ndim // 2, a.ndim)))
    c = 0
    for perm in itertools.permutations(range(b.ndim // 2, b.ndim)):
        c += b.transpose(tuple(range(b.ndim // 2)) + perm)
    return c / (np.math.factorial(b.ndim // 2)) ** 2


def permanent_old(covariance):
    cov = covariance
    cov2 = symmetrize(2 * np.tensordot(cov, cov, axes=0).transpose((0, 2, 1, 3)))

    while cov.size > 1:
        n = cov2.shape[0]
        cov4 = symmetrize(
            18 * np.tensordot(cov2, cov2, axes=0).transpose((0, 1, 4, 5, 2, 3, 6, 7))
            - 48
            * np.tensordot(
                np.tensordot(cov, cov, axes=0),
                np.tensordot(cov, cov, axes=0),
                axes=0,
            ).transpose((0, 2, 4, 6, 1, 3, 5, 7))
        )
        cov = cov2[np.arange(0, n, 2), np.arange(1, n, 2), :, :][
            :, np.arange(0, n, 2), np.arange(1, n, 2)
        ]
        cov2 = cov4[np.arange(0, n, 2), np.arange(1, n, 2), :, :, :, :, :, :][
            :, np.arange(0, n, 2), np.arange(1, n, 2), :, :, :, :
        ][:, :, np.arange(0, n, 2), np.arange(1, n, 2), :, :][
            :, :, :, np.arange(0, n, 2), np.arange(1, n, 2)
        ]

    return cov.item()


import numpy as np
import itertools


def symmetrize_left(a):
    assert a.ndim % 2 == 0
    result = 0
    for perm in itertools.permutations(range(a.ndim // 2)):
        result += a.transpose(perm + tuple(range(a.ndim // 2, a.ndim)))
    return result / np.math.factorial(a.ndim // 2)


def symmetrize_right(a):
    assert a.ndim % 2 == 0
    result = 0
    for perm in itertools.permutations(range(a.ndim // 2, a.ndim)):
        result += a.transpose(tuple(range(a.ndim // 2)) + perm)
    return result / np.math.factorial(a.ndim // 2)


def next_level(cov):
    n = cov.shape[0]
    return cov[np.arange(0, n, 2), np.arange(1, n, 2), :, :, :, :, :, :][
        :, np.arange(0, n, 2), np.arange(1, n, 2), :, :, :, :
    ][:, :, np.arange(0, n, 2), np.arange(1, n, 2), :, :][
        :, :, :, np.arange(0, n, 2), np.arange(1, n, 2)
    ]


def symmetric_power(mat, exp):
    n = mat.ndim
    power = functools.reduce(lambda a, b: np.tensordot(a, b, axes=0), (mat,) * exp)
    power = power.transpose(tuple(i + j * n for i in range(n) for j in range(exp)))
    for dim in range(n):
        accum = 0
        for perm in itertools.permutations(range(dim * exp, (dim + 1) * exp)):
            accum += power.transpose(
                tuple(range(dim * exp)) + perm + tuple(range((dim + 1) * exp, n * exp))
            )
        power = accum / np.math.factorial(exp)
    return power


def permanent(covariance):
    cov = covariance
    cov2 = 2 * symmetric_power(cov, 2)

    while cov.size > 1:
        n = cov2.shape[0]
        cov4 = symmetrize_left(
            symmetrize_right(
                18
                * np.tensordot(cov2, cov2, axes=0).transpose((0, 1, 4, 5, 2, 3, 6, 7))
            )
        ) - 48 * symmetric_power(cov, 4)
        cov = cov2[np.arange(0, n, 2), np.arange(1, n, 2), :, :][
            :, np.arange(0, n, 2), np.arange(1, n, 2)
        ]
        cov2 = cov4[np.arange(0, n, 2), np.arange(1, n, 2), :, :, :, :, :, :][
            :, np.arange(0, n, 2), np.arange(1, n, 2), :, :, :, :
        ][:, :, np.arange(0, n, 2), np.arange(1, n, 2), :, :][
            :, :, :, np.arange(0, n, 2), np.arange(1, n, 2)
        ]

    return cov.item()


def symmetric_left_power(mat, exp):
    n = mat.ndim
    power = functools.reduce(lambda a, b: np.tensordot(a, b, axes=0), (mat,) * exp)
    power = power.transpose(tuple(i + j * n for i in range(n) for j in range(exp)))
    power = power.reshape(power.shape[: (n - 1) * exp] + (-1,))
    result = 0
    for perm in itertools.permutations(range((n - 1) * exp)):
        result += power.transpose(perm + ((n - 1) * exp,))
    result = result / np.math.factorial((n - 1) * exp) ** 0.5
    return result


def symmetric_left_power2(mat, exp):
    n = mat.ndim
    power = functools.reduce(lambda a, b: np.tensordot(a, b, axes=0), (mat,) * exp)
    power = power.transpose(tuple(i + j * n for i in range(n) for j in range(exp)))
    power = power.reshape(power.shape[: (n - 1) * exp] + (-1,))
    result = 0
    # for perm in [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (1, 0, 2, 3), (1, 0, 3, 2), (3, 1, 0, 2)]:
    #     result += power.transpose(perm + ((n - 1) * exp,))
    for perm in [(0, 1, 2, 3), (1, 0, 3, 2)]:
        result += power.transpose(perm + ((n - 1) * exp,))
    power = result
    result = 0
    for perm in [
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 2, 1, 3),
    ]:  # [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]:
        result += power.transpose(perm + ((n - 1) * exp,))
    result = result / 6**0.5  # np.math.factorial((n - 1) * exp)
    return result


def split(cov):
    n1, n2 = np.prod(cov.shape[: cov.ndim // 2]), np.prod(cov.shape[cov.ndim // 2 :])
    U, S, Vh = np.linalg.svd(cov.reshape(n1, n2))
    u = (U @ np.diag(S**0.5)).reshape(cov.shape[: cov.ndim // 2] + (-1,))
    v = (Vh.T @ np.diag(S**0.5)).reshape(cov.shape[cov.ndim // 2 :] + (-1,))
    return u, v


def project(u, v):
    cov = np.tensordot(u, v, axes=((-1), (-1)))
    return split(cov)


def factored_permanent(covariance):
    cov = covariance
    u, v = split(cov)
    u2 = symmetric_left_power(u, 2)
    v2 = symmetric_left_power(v, 2)

    while u.size > 1:
        n = u.shape[0]

        u4 = np.concatenate(
            [
                3**0.5 * symmetric_left_power2(u2, 2),
                2**0.5 * 1j * symmetric_left_power2(symmetric_left_power(u, 2), 2),
            ],
            axis=-1,
        )
        v4 = np.concatenate(
            [
                3**0.5 * symmetric_left_power2(v2, 2),
                2**0.5 * 1j * symmetric_left_power2(symmetric_left_power(v, 2), 2),
            ],
            axis=-1,
        )

        u = u2[np.arange(0, n, 2), np.arange(1, n, 2), :]
        v = v2[np.arange(0, n, 2), np.arange(1, n, 2), :]
        u2 = u4[np.arange(0, n, 2), np.arange(1, n, 2), :, :, :][
            :, np.arange(0, n, 2), np.arange(1, n, 2), :
        ]
        v2 = v4[np.arange(0, n, 2), np.arange(1, n, 2), :, :, :][
            :, np.arange(0, n, 2), np.arange(1, n, 2), :
        ]

        u, v = project(u, v)
        u2, v2 = project(u2, v2)

    return np.real(u @ v.transpose()).item()
