from typing import Set, List, Optional, Counter

from treelib import Tree

from exploration.baby_giant_step import baby_giant_step
from exploration.picaboo import squeeze_at_root
from exploration.tree_cover import tree_cover_dfs
from trees.configuration import Configuration, find_root
from trees.traversal import Traversal


def dfs_explore(tree: Tree,
                num_robots: int,
                start_config: Optional[Configuration] = None) -> Traversal:
    if not start_config:
        start_config = {tree.root: num_robots}
    current_config = start_config

    # Perform a baby-giant step
    traversal = list(baby_giant_step(tree, num_robots, start_config=start_config))
    # Stop condition: if baby-giant covered the subtree we can recursively return traversal
    if len(tree.nodes) <= 1:
        return tuple(traversal)

    # tree-cover the visited subtree
    covering = tree_cover_dfs(Tree(tree, deep=True), num_robots//4)

    # Go in dfs visit and explore each subtree in the cover recursively
    for cov_subtree in covering:
        # Update subtree as some vertices might have been already removed
        subtree = {v for v in cov_subtree if v in tree.nodes}
        subtree_root = find_root({v: 1 for v in subtree}, tree)
        # Go to root
        go_to_root = to_subtree_root(tree, num_robots, subtree_root, current_config)
        traversal.extend(list(go_to_root))
        # Recursively traverse the subtree
        dfs_recursive_exploration = dfs_explore(get_subtree(tree, subtree), num_robots)
        traversal.extend(list(dfs_recursive_exploration))
        # Squeeze at root
        squeeze = squeeze_at_root(tree, traversal[-1])
        traversal.extend(list(squeeze))
        current_config = traversal[-1]
        # Gather back at subtree_root
        gathering = to_subtree_root(tree, num_robots, subtree_root, current_config)
        traversal.extend(list(gathering))
    return tuple(traversal)


def to_subtree_root(tree: Tree,
                    num_robots: int,
                    subtree_root: str,
                    start_config: Optional[Configuration] = None) -> Traversal:
    if not start_config:
        start_config = {tree.root: num_robots}
    else:
        assert len(start_config.keys()) == 1
        assert list(start_config.values())[0] == num_robots
    start_node = list(start_config.keys())[0]
    path_to_root = find_path(tree, start_node, subtree_root)
    return tuple({v: num_robots} for v in path_to_root)


def find_path(tree, start_id, end_id):
    """Find path between two nodes."""
    if start_id == end_id:
        return []

    # Get path from start to root
    start_path = list(tree.rsearch(start_id))
    # Get path from end to root
    end_path = list(tree.rsearch(end_id))

    # Find common ancestor
    common_ancestors = set(start_path) & set(end_path)
    if not common_ancestors:
        return None

    # Find lowest common ancestor
    lca = min(common_ancestors, key=lambda x: tree.level(x))

    # Construct path
    start_to_lca = start_path[:start_path.index(lca)]
    lca_to_end = end_path[:end_path.index(lca)]

    return start_to_lca + [lca] + lca_to_end[::-1]


def get_subtree(tree: Tree, subset: Set[str]) -> Tree:
    subtree_root = find_root({v: 1 for v in subset}, tree)
    subtree = Tree()
    subtree.create_node(identifier=subtree_root, tag=tree[subtree_root].tag)

    for v in tree.expand_tree(nid=subtree_root, mode=Tree.DEPTH, reverse=True):
        if v == subtree_root:
            continue
        if set(tree.rsearch(v)) & subset:
            subtree.create_node(identifier=v, tag=tree[v].tag, parent=tree.parent(v))

    return subtree
