from collections import deque


def bill_subtree_indices(tree_rep):
    """Return index paths of nested list subtrees in breadth-first order."""
    q = deque([([], tree_rep)])
    out = []

    while q:
        indices, subtree = q.popleft()
        out.append(indices)
        for ordinal, sst in enumerate(subtree[1:]):
            if isinstance(sst, list):
                idxs = indices[:]
                idxs.append(ordinal + 1)
                q.append((idxs, sst))

    return out


def get_subtree(tree_rep, index_path):
    """Follow index path into nested list tree and return subtree node."""
    node = tree_rep
    for idx in index_path:
        node = node[idx]
    return node
