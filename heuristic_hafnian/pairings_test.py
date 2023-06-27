from .pairings import all_pairings


def test_all_pairings():
    counts = [len(list(all_pairings(range(n)))) for n in range(0, 12, 2)]
    assert counts == [1, 1, 3, 15, 105, 945]
