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
    if not isinstance(btree, Iterable):
        return (btree,)
    return tuple(item for subtree in btree for item in flatten_btree(subtree))


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
    if not isinstance(btree, Iterable):
        return {btree: 0}
    heights = {}
    for subtree in btree:
        heights = {**heights, **btree_heights(subtree)}
    heights[btree] = max(heights.values()) + 1
    return heights


def btree_sides(btree):
    sides = {}
    for side in (0, 1):
        for level in btree_levels(btree[side]):
            for subtree in level:
                sides[subtree] = side
    return sides
