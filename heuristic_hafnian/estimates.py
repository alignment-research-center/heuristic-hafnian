"""
Heuristic estimates for the permanent.
"""

import itertools

import numpy as np
from scipy.special import comb, factorial, factorial2

from .propagation import cumulant_propagation


def est_row(mat):
    n = mat.shape[0]
    assert mat.shape == (n, n)
    return mat.sum(1).prod() * factorial(n) / n**n


def est_col(mat):
    n = mat.shape[0]
    assert mat.shape == (n, n)
    return mat.sum(0).prod() * factorial(n) / n**n


def est_sum(mat):
    n = mat.shape[0]
    assert mat.shape == (n, n)
    return mat.sum() ** n * factorial(n) / n ** (2 * n)


def est_uniq(mat):
    n = mat.shape[0]
    assert mat.shape == (n, n)
    values, counts = np.unique(mat, return_counts=True)
    k = len(values)
    total = 0.0
    for bars in itertools.combinations(range(n + k - 1), k - 1):
        stars = [b - a - 1 for a, b in zip((-1,) + bars, bars + (n + k - 1,))]
        total += np.prod(
            [v**s * comb(c, s) for v, c, s in zip(values, counts, stars)]
        )
    return total * factorial(n) / comb(n**2, n)


def est_row_col_sum(mat):
    denom = est_sum(mat)
    if denom == 0:
        return 0.0
    return est_row(mat) * est_col(mat) / est_sum(mat)


def est_row_col_uniq(mat):
    denom = est_uniq(mat)
    if denom == 0:
        return 0.0
    return est_row(mat) * est_col(mat) / est_uniq(mat)


def est_norm(mat):
    n = mat.shape[0]
    assert mat.shape == (n, n)
    mu = mat.sum() / n**2
    mu2 = (
        sum(
            mat[i, j] * np.sum(mat[np.arange(n) != i, :][:, np.arange(n) != j])
            for i, j in itertools.product(*(range(n),) * 2)
        )
        / (n * (n - 1)) ** 2
        if n > 1
        else float("nan")
    )
    total = mu**n + sum(
        comb(n, 2 * k)
        * factorial2(2 * k - 1)
        * (mu2 - mu**2) ** k
        * mu ** (n - 2 * k)
        for k in range(1, n // 2 + 1)
    )
    return total * factorial(n)


def zero_block_diag(mat):
    """
    Returns the block matrix ((0 mat) (mat^T 0)),
    whose hafnian is the permanent of mat.
    """
    return np.block(
        [[np.zeros(mat.shape), mat], [mat.transpose(), np.zeros(mat.shape)]]
    )


def est_cov(mat):
    covariance = zero_block_diag(mat)
    return cumulant_propagation(covariance, order=3)
