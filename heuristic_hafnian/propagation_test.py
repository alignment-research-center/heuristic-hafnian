import numpy as np
from thewalrus.reference import hafnian

from .propagation import (
    connected_partitions,
    cumulant_propagation,
    extended_covariance_propagation_v1,
    extended_covariance_propagation_v2,
)


def test_connected_partitions():
    counts = [
        len(list(connected_partitions([(2 * i, 2 * i + 1) for i in range(n)])))
        for n in range(1, 5)
    ]
    assert counts == [2, 11, 129, 2465]


def test_cumulant_propagation():
    np.random.seed(0)
    for n in range(1, 9):
        mat = np.random.randn(n, n)
        covariance = mat @ mat.transpose()
        result = cumulant_propagation(covariance)
        assert np.isclose(result, hafnian(covariance))


def test_extended_covariance_propagation():
    np.random.seed(0)
    for n in range(2, 13, 2):
        mat = np.random.randn(n, n)
        covariance = mat @ mat.transpose()
        result_v1 = extended_covariance_propagation_v1(covariance)
        result_v2 = extended_covariance_propagation_v2(covariance)
        assert np.isclose(result_v1, result_v2)