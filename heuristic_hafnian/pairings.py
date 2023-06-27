import numpy as np


def all_pairings(items):
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
    items = sorted([item for pair in pairing for item in pair])
    n = len(items)
    offset = items[0]
    assert items == list(range(offset, offset + n))
    matrix = np.zeros((n, n))
    for i, j in pairing:
        matrix[i - offset, j - offset] = 1
        matrix[j - offset, i - offset] = 1
    return matrix
