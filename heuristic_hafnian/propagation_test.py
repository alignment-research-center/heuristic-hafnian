import numpy as np
from thewalrus.reference import hafnian

from .propagation import connected_partitions, cumulant_propagation


def test_connected_partitions():
    counts = [
        len(list(connected_partitions([(2 * i, 2 * i + 1) for i in range(n)])))
        for n in range(1, 5)
    ]
    assert counts == [2, 11, 129, 2465]


def test_cumulant_propagation():
    np.random.seed(0)
    for n in range(2, 10, 2):
        mat = np.random.randn(n, n)
        covariance = mat @ mat.transpose()
        result = cumulant_propagation(covariance, order=4)
        assert np.isclose(result, hafnian(covariance))
