import itertools

import numpy as np
from scipy.special import comb, factorial

from .estimates import est_uniq


def est_uniq_slow(mat):
    n = mat.shape[0]
    assert mat.shape == (n, n)
    total = 0.0
    for indices in itertools.combinations(itertools.product(*(range(n),) * 2), n):
        total += np.prod([mat[i] for i in indices])
    return total * factorial(n) / comb(n**2, n)


def test_est_uniq():
    np.random.seed(0)
    for n in range(1, 5):
        mat = np.random.randint(2, size=n**2).reshape(n, n)
        assert est_uniq(mat) == est_uniq_slow(mat)
