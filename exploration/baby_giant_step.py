from collections import Counter
from itertools import zip_longest
from math import ceil, sqrt
from typing import Optional, Set

from treelib import Tree

from exploration.picaboo import squeeze_at_root, picaboo
from trees.configuration import Configuration, find_root
from trees.table import fpt_compute_traversal
from trees.traversal import Traversal


def expand_to_dangling_leaves(tree: Tree, start_config: Configuration, leaves: Set[str]) -> Traversal:
    assert len(start_config.keys()) == 1
    current_config = start_config
    num_robots = sum(start_config.values())
    root = find_root(current_config, tree)
    traversal = []

    # filter leaves that were deleted
    leaves = {leaf for leaf in leaves if leaf in tree.nodes}

    internal_subtree = {ancestor for leaf in leaves for ancestor in tree.rsearch(leaf)} - leaves
    internal_subtree = internal_subtree - (set(tree.rsearch(root)) - {root})  # no need to occupy above current root
    if len(leaves) == 0:
        return ()

    leaf_budget = num_robots - len(internal_subtree)
    per_leaf_budget = leaf_budget // len(leaves)
    remainder = leaf_budget % len(leaves)
    depth = max(tree.depth(leaf) - tree.depth(root) for leaf in leaves)

    for d in range(1, depth+1):
        remainder_budget = remainder
        next_config = Counter()
        current_internal = [nid for nid in tree.expand_tree(nid=root, mode=Tree.WIDTH)
                            if tree.depth(nid) - tree.depth(root) < d]
        current_internal = [nid for nid in current_internal if nid in internal_subtree | leaves]

        current_frontier = [nid for nid in tree.expand_tree(nid=root, mode=Tree.WIDTH)
                            if tree.depth(nid) - tree.depth(root) == d]
        current_frontier = [nid for nid in current_frontier if nid in internal_subtree | leaves]
        for internal_node in current_internal:
            if internal_node in leaves:
                next_config[internal_node] = current_config[internal_node]  # wait at the leaf
                remainder_budget -= current_config[internal_node] > per_leaf_budget  # consume remainder again
            else:
                next_config[internal_node] = 1  # wait for connectivity
        for j, leaf in enumerate(current_frontier):
            interior_scan = [nid for nid in tree.expand_tree(nid=leaf, mode=Tree.WIDTH)
                             if nid in internal_subtree]  # robots to send for connectivity below current leaf frontier
            leaf_detected = [nid for nid in tree.expand_tree(nid=leaf, mode=Tree.WIDTH)
                             if nid in leaves]  # leaves for exploration
            next_config[leaf] = len(interior_scan) + per_leaf_budget * len(leaf_detected)
            # Take care of remainder
            from_remainder = min(remainder_budget, len(leaf_detected))
            next_config[leaf] += from_remainder
            remainder_budget -= from_remainder  # only consume the remainder at the actual leaves
        assert sum(current_config.values()) == sum(next_config.values()), f"Number of robots must be preserved"
        current_config = next_config.copy()
        traversal.append(current_config)
    return tuple(traversal)

def baby_giant_step(tree: Tree,
                    num_robots: int,
                    max_depth: Optional[int] = None,
                    start_config: Optional[Configuration] = None) -> Traversal:
    if max_depth:
        assert max_depth >= 0
    else:
        max_depth = ceil(sqrt(num_robots))
    picaboo_counter = ceil(num_robots / max_depth)

    if num_robots == 1:
        return fpt_compute_traversal(tree, 1)

    if not start_config:
        current_config = Counter()
        current_config[tree.root] = num_robots
    else:
        assert sum(start_config.values()) == num_robots
        current_config = start_config

    squeezing = squeeze_at_root(tree, current_config)
    traversal = [current_config] + list(squeezing)
    leaves = {find_root(current_config, tree)}
    for t in range(2*picaboo_counter):
        # Baby:
        # Group all robots at r
        gathering = squeeze_at_root(tree, traversal[-1])
        traversal.extend(list(gathering))
        # Update leaves as we might have cleared some
        leaves = {leaf for leaf in leaves if leaf in tree.nodes and len(tree.children(leaf)) > 0}
        # Expand to dangling leaves equally
        to_dangling = expand_to_dangling_leaves(tree, traversal[-1], leaves)
        traversal.extend(list(to_dangling))

        # Giant:
        # Call Picaboo in parallel on all leaves
        parallel_picaboos = dict()
        gathered_leaves = set()
        for leaf in leaves:
            res = picaboo(tree, traversal[-1][leaf], max_depth=max_depth, start_config=Counter({leaf: traversal[-1][leaf]}))
            if res:
                picaboo_session, discovered_leaves = res
                gathered_leaves |= discovered_leaves
                parallel_picaboos[leaf] = picaboo_session
        picaboo_len = max([0] + [len(picaboo_session) for picaboo_session in parallel_picaboos.values()])
        padded_parallel_picaboos = []
        # If other picaboos end before, wait at their leaves for the rest to finish
        for leaf, picaboo_session in parallel_picaboos.items():
            padded_picaboo_session = list(picaboo_session)
            padded_picaboo_session.extend((picaboo_len - len(picaboo_session)) * [{leaf: traversal[-1][leaf]}])
            padded_parallel_picaboos.append(padded_picaboo_session)
        zipped_picaboos = list(zip(*padded_parallel_picaboos))
        # Add the static robots that keep connectivity
        static_robots = {node: 1 for node in traversal[-1] if node not in leaves}
        parallel_picaboo = [{node: count for conf in parallel_confs for node, count in conf.items()} | static_robots
                            for parallel_confs in zipped_picaboos]

        leaves = gathered_leaves
        traversal.extend(parallel_picaboo)

    return tuple(traversal)
