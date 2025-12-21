from collections import Counter
from math import ceil, sqrt

from typing import Optional, Set
from enum import Enum
from treelib import Tree

from trees.configuration import find_root, Configuration
from trees.table import fpt_compute_traversal
from trees.traversal import Traversal

class NodeState(Enum):
    UNVISITED = 1
    VISITED = 2
    FINISHED = 3


def squeeze_at_root(tree: Tree, start_config: Configuration) -> Traversal:
    num_robots = sum(start_config.values())
    current_config = {nid: count for nid, count in start_config.items() if nid in tree.nodes}
    root = find_root(current_config, tree)
    traversal = []

    if root not in tree.nodes:
        print('here')

    while current_config[root] < num_robots:
        next_config = Counter()
        reverse_dfs = [nid for nid in tree.expand_tree(nid=root, mode=Tree.WIDTH, reverse=True)
                       if nid in current_config.keys()]
        for node in reverse_dfs:
            if node == root:
                next_config[root] += current_config[root]
            else:
                parent = tree.parent(node).identifier
                next_config[parent] = next_config.get(parent, 0) + current_config[node]

                # Also, remove node if it has no children as we don't want to revisit it
                # This is why we scan in reverse BFS
                if len(tree.children(node)) == 0:
                    tree.remove_node(node)

        current_config = next_config.copy()
        traversal.append(current_config)
        if root not in tree.nodes:
            print('here')
    return tuple(traversal)


def expand_from_root(tree: Tree, start_config: Configuration, depth: int) -> (Traversal, Set[str]):
    assert len(start_config.keys()) == 1
    current_config = start_config
    num_robots = sum(start_config.values())
    root = find_root(current_config, tree)
    traversal = []

    internal_subtree = [nid for nid in tree.expand_tree(nid=root, mode=Tree.WIDTH)
                        if tree.depth(nid) - tree.depth(root) < depth]
    subtree_leaves = {nid for nid in tree.expand_tree(nid=root, mode=Tree.WIDTH)
                      if tree.depth(nid) - tree.depth(root) == depth}

    if len(subtree_leaves) == 0:
        return tuple(), set()
    if len(internal_subtree) + len(subtree_leaves) > num_robots:
        return

    leaf_budget = num_robots - len(internal_subtree)
    per_leaf_budget = leaf_budget // len(subtree_leaves)

    for d in range(1, depth+1):
        next_config = Counter()
        remainder = leaf_budget % len(subtree_leaves)
        internal_scan = [nid for nid in tree.expand_tree(nid=root, mode=Tree.WIDTH)
                         if tree.depth(nid) - tree.depth(root) < d]
        leaf_scan = [nid for nid in tree.expand_tree(nid=root, mode=Tree.WIDTH)
                     if tree.depth(nid) - tree.depth(root) == d]
        for internal_node in internal_scan:
            next_config[internal_node] = 1
        for j, leaf in enumerate(leaf_scan):
            interior_scan = [nid for nid in tree.expand_tree(nid=leaf, mode=Tree.WIDTH)
                             if nid in internal_subtree]
            leaf_detected = [nid for nid in tree.expand_tree(nid=leaf, mode=Tree.WIDTH)
                         if nid in subtree_leaves]
            next_config[leaf] = len(interior_scan) + per_leaf_budget * len(leaf_detected)
            # Take care of remainder
            next_config[leaf] += min(remainder, len(leaf_detected))
            remainder -= min(remainder, len(leaf_detected))
        current_config = next_config.copy()
        traversal.append(current_config)
    return tuple(traversal), subtree_leaves


def picaboo(tree: Tree,
            num_robots: int,
            max_depth: Optional[int] = None,
            start_config: Optional[Configuration] = None) -> (Traversal, Set[str]):
    if max_depth:
        assert max_depth >= 0
    else:
        max_depth = ceil(sqrt(num_robots))

    if num_robots == 1:
        return tuple(), set()

    if not start_config:
        current_config = Counter()
        current_config[tree.root] = num_robots
    else:
        assert sum(start_config.values()) == num_robots
        current_config = start_config

    root = find_root(current_config, tree)
    current_config = {root: num_robots}
    leaves = {root}
    traversal = [current_config] + list(squeeze_at_root(tree, current_config))
    for depth in range(1, max_depth+1):
        # Pica:
        res = expand_from_root(tree, current_config, depth)
        if res:
            expansion, leaves = res
            traversal.extend(list(expansion))
        # Boo:
        gathering = squeeze_at_root(tree, traversal[-1])
        traversal.extend(list(gathering))
        # After gathering we are back at current_config with all robots at root
    return tuple(traversal), leaves




