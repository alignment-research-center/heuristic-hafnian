import numpy as np

from .permutations import (
    all_permutations,
    indicator_of_permutation,
    product_over_permutation,
    random_permutation,
)


def test_all_permutations():
    counts = [len(list(all_permutations(range(n)))) for n in range(0, 7)]
    assert counts == [1, 1, 2, 6, 24, 120, 720]


def test_indicator_of_permutation():
    np.random.seed(0)
    for n in range(2, 7):
        first_permutation = random_permutation(range(n))
        second_permutation = first_permutation
        while second_permutation == first_permutation:
            second_permutation = random_permutation(range(n))
        mat = indicator_of_permutation(first_permutation)
        assert product_over_permutation(mat, first_permutation) == 1
        assert product_over_permutation(mat, second_permutation) == 0
