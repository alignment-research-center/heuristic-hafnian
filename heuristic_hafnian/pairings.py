import numpy as np


def all_pairings(items):
    """
    Yields all pairings (i.e., partitions in which each part
    has size 2) of the given items.
    """
    items = list(items)
    if len(items) == 0:
        yield []
        return

    first_item = items.pop(0)
    for i, item in enumerate(items):
        first_pair = (first_item, item)
        for pairing in all_pairings(items[:i] + items[i + 1 :]):
            yield [first_pair] + pairing


def random_pairing(items):
    """
    Returns a uniformly random pairing (i.e., partition in which
    each part has size 2) of the given items.
    """
    items = list(items)
    assert len(items) % 2 == 0
    if len(items) == 0:
        return []

    first_item = items.pop(0)
    i = np.random.randint(0, len(items))
    first_pair = (first_item, items[i])
    pairing = random_pairing(items[:i] + items[i + 1 :])
    return [first_pair] + pairing


def product_over_pairing(matrix, pairing):
    """
    Given a symmetric matrix, returns the contribution to the hafnian
    for the given pairing, i.e. the product of the (i, j)th entries
    of the matrix for each (i, j) in the pairing.
    """
    items = sorted([item for pair in pairing for item in pair])
    n = len(items)
    offset = items[0]
    assert items == list(range(offset, offset + n))
    result = 1
    for i, j in pairing:
        result1 = result * matrix[i - offset, j - offset]
        result2 = result * matrix[j - offset, i - offset]
        assert np.isclose(result1, result2)
        result = result1
    return result


def elementary_matrix(pairing):
    """
    Given a pairing, returns the symmetric matrix with whose (i, j)th
    entry is 1 if (i, j) or (j, i) is in the pairing and 0 otherwise.
    """
    items = sorted([item for pair in pairing for item in pair])
    n = len(items)
    offset = items[0]
    assert items == list(range(offset, offset + n))
    matrix = np.zeros((n, n))
    for i, j in pairing:
        matrix[i - offset, j - offset] = 1
        matrix[j - offset, i - offset] = 1
    return matrix
