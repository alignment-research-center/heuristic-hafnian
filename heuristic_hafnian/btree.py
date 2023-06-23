from collections.abc import Iterable


def balanced_btree(items):
    if len(items) == 1:
        return items[0]
    mid = (len(items) + 1) // 2
    return (balanced_btree(items[:mid]), balanced_btree(items[mid:]))


def unbalanced_btree(items):
    if len(items) == 1:
        return items[0]
    return (unbalanced_btree(items[:1]), unbalanced_btree(items[1:]))


def flatten_btree(btree):
    if isinstance(btree, Iterable):
        return tuple(item for subtree in btree for item in flatten_btree(subtree))
    else:
        return (btree,)


def btree_levels(btree):
    btree = (btree,)
    yield btree
    while any(isinstance(subtree, Iterable) for subtree in btree):
        btree = tuple(
            item
            for subtree in btree
            if isinstance(subtree, Iterable)
            for item in subtree
        )
        yield btree


def btree_heights(btree):
    heights = {}
    for height, level in enumerate(reversed(list(btree_levels(btree)))):
        for subtree in level:
            heights[subtree] = height
    return heights


def btree_sides(btree):
    sides = {}
    for side in (0, 1):
        for level in btree_levels(btree[side]):
            for subtree in level:
                sides[subtree] = side
    return sides


