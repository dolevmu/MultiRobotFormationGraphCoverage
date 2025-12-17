from typing import Set, List

from treelib import Tree

from trees.configuration import find_root


def tree_cover_dfs(tree: Tree, min_subtree_size: int) -> List[Set[str]]:
    # Covers the input tree with a collocation of trees of sizes between [L,U)
    # where L=min_subtree_size and U=2L, except for one tree rooted at the root
    # of size <= L.
    # This one works for the DFS algorithm.

    tree_collections = list()
    subtree_size_mapping = dict()
    dfs_scan = list(tree.expand_tree(mode=Tree.DEPTH))

    for node in reversed(dfs_scan):
        sorted_children = sorted(tree.children(node), key=tree.depth, reverse=True)

        subtree_size_mapping[node] = 1
        subtree = {node}
        for u in sorted_children:
            subtree_size_mapping[node] += subtree_size_mapping[u.identifier]
            subtree |= set(tree.subtree(u.identifier).nodes.keys())
            if subtree_size_mapping[node] >= min_subtree_size:
                tree_collections.append(subtree)
                subtree = {node}
                subtree_size_mapping[node] = 1
                tree.remove_subtree(u.identifier)
    if len(tree.nodes) > 1:
        tree_collections.append(set(tree.nodes.keys()))
    return tree_collections


def parse_tree_cover(tree: Tree, covering: List[Set[str]]) -> Tree:
    covering_tree = Tree()
    covering_roots = {find_root(conf, tree): conf for conf in covering_tree}


