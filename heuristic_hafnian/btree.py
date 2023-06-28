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


def breadth_first(btree):
    for level in btree_levels(btree):
        for subtree in level:
            yield subtree


def depth_first(btree):
    yield btree
    if not isinstance(btree, Iterable):
        return
    for subtree in btree:
        yield from depth_first(subtree)


def btree_height(btree):
    if not isinstance(btree, Iterable):
        return 0
    return max(btree_height(subtree) for subtree in btree) + 1
