import numpy as np

from .pairings import (
    all_pairings,
    indicator_of_pairing,
    product_over_pairing,
    random_pairing,
)


def test_all_pairings():
    counts = [len(list(all_pairings(range(n)))) for n in range(0, 12, 2)]
    assert counts == [1, 1, 3, 15, 105, 945]


def test_indicator_of_pairing():
    np.random.seed(0)
    for n in range(4, 12, 2):
        first_pairing = random_pairing(range(n))
        second_pairing = first_pairing
        while second_pairing == first_pairing:
            second_pairing = random_pairing(range(n))
        mat = indicator_of_pairing(first_pairing)
        assert product_over_pairing(mat, first_pairing) == 1
        assert product_over_pairing(mat, second_pairing) == 0
