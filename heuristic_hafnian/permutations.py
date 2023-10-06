import numpy as np


def all_permutations(items):
    """
    Yields all permutations of the given items.
    """
    items = list(items)
    if len(items) == 0:
        yield []
        return

    for i, item in enumerate(items):
        for permutation in all_permutations(items[:i] + items[i + 1 :]):
            yield [item] + permutation


def random_permutation(items):
    """
    Returns a uniformly random permutation of the given items.
    """
    items = list(items)
    if len(items) == 0:
        return []

    i = np.random.randint(0, len(items))
    item = items[i]
    permutation = random_permutation(items[:i] + items[i + 1 :])
    return [item] + permutation


def product_over_permutation(matrix, permutation):
    """
    Given a matrix, returns the contribution to the permanent for the
    given permutation, i.e. the product of the (i, permutation(i))th
    entries of the matrix for each i.
    """
    items = sorted(permutation)
    n = len(items)
    offset = items[0]
    assert items == list(range(offset, offset + n))
    result = 1
    for i in range(n):
        result = result * matrix[i, permutation[i] - offset]
    return result


def indicator_of_permutation(permutation):
    """
    Given a permutation, returns the matrix with whose (i, j)th entry
    is 1 if j = permutation(i) and 0 otherwise.
    """
    items = sorted(permutation)
    n = len(items)
    offset = items[0]
    assert items == list(range(offset, offset + n))
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, permutation[i] - offset] = 1
    return matrix
