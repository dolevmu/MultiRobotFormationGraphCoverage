from typing import Set, List, Optional, Counter, Tuple
from math import sqrt, ceil

from treelib import Tree

from exploration.baby_giant_step import baby_giant_step
from exploration.picaboo import squeeze_at_root, picaboo
from exploration.optimistic_bgs import optimistic_bgs
from exploration.tree_cover import tree_cover_dfs
from trees.configuration import Configuration, find_root
from trees.traversal import Traversal


def dfs_explore(tree: Tree,
                num_robots: int,
                start_config: Optional[Configuration] = None,
                depth_constant: float = 1.0,
                step_algorithm: str = "bgs",
                max_depth: Optional[int] = None) -> Traversal:
    """
    DFS-based tree exploration algorithm.

    Args:
        step_algorithm: "bgs" for baby-giant-step, "picaboo" for picaboo algorithm
        max_depth: Maximum depth for baby-giant-step (delta). If None, uses sqrt(num_robots).
    """
    traversal, _ = dfs_explore_internal(Tree(tree, deep=True), num_robots, start_config=start_config,
                                        depth_constant=depth_constant, step_algorithm=step_algorithm,
                                        max_depth=max_depth)

    if len(traversal) == 0:
        return traversal

    exploration_traversal = [traversal[0]]
    for cfg in traversal[1:]:
        if cfg != exploration_traversal[-1]:
            exploration_traversal.append(cfg)
    return tuple(exploration_traversal)


def dfs_explore_internal(tree: Tree,
                         num_robots: int,
                         start_config: Optional[Configuration] = None,
                         depth_constant: float = 1.0,
                         step_algorithm: str = "bgs",
                         max_depth: Optional[int] = None) -> Tuple[Traversal, Set[str]]:
    vertices = set(tree.nodes.keys())
    if not start_config:
        start_config = {tree.root: num_robots}
    current_config = start_config

    # Perform exploration step (baby-giant-step, picaboo, or optimistic_bgs)
    if step_algorithm == "bgs":
        traversal = list(baby_giant_step(tree, num_robots, max_depth=max_depth, start_config=start_config))
    elif step_algorithm == "picaboo":
        # Picaboo requires enough robots to cover internal nodes + leaves at each depth.
        # For trees with high branching factor, picaboo struggles even if it can technically
        # start. Use a conservative check: require enough robots to cover all nodes up to
        # max_depth (delta), which is the depth picaboo will try to explore.
        root = find_root(start_config, tree)

        # Count nodes at each depth level up to max_depth
        effective_max_depth = max_depth if max_depth else ceil(sqrt(num_robots))
        nodes_to_cover = 0
        current_level = [root]
        for d in range(effective_max_depth + 1):
            nodes_to_cover += len(current_level)
            next_level = []
            for node in current_level:
                next_level.extend([c.identifier for c in tree.children(node)])
            current_level = next_level
            if not current_level:
                break

        # Use picaboo only if we can cover all nodes up to max_depth
        if nodes_to_cover <= num_robots:
            result = picaboo(tree, num_robots, max_depth=max_depth, start_config=start_config)
            if result:
                traversal = list(result[0])
            else:
                traversal = list(baby_giant_step(tree, num_robots, max_depth=max_depth, start_config=start_config))
        else:
            # Not enough robots for picaboo - use BGS which handles high-degree better
            traversal = list(baby_giant_step(tree, num_robots, max_depth=max_depth, start_config=start_config))
    elif step_algorithm == "optimistic":
        result, _ = optimistic_bgs(tree, num_robots, start_config=start_config)
        traversal = list(result)
    else:
        raise ValueError(f"Unknown step_algorithm: {step_algorithm}")

    # Ensure traversal is never empty
    if len(traversal) == 0:
        traversal = [start_config]

    # Update current_config to the last position in traversal
    current_config = traversal[-1]

    # Stop condition: if baby-giant covered the subtree we can recursively return traversal
    if len(tree.nodes) <= 1:
        deleted = vertices - set(tree.nodes.keys())
        return tuple(traversal), deleted

    # Gather all robots before dividing into subtrees
    # (step algorithm might leave robots distributed across multiple nodes)
    if len(current_config) > 1:
        gathering = squeeze_at_root(tree, current_config)
        traversal.extend(list(gathering))
        if len(traversal) > 0:
            current_config = traversal[-1]

    # Check remaining tree depth vs C * sqrt(k)
    remaining_depth = tree.depth()
    depth_threshold = depth_constant * sqrt(num_robots)

    if remaining_depth <= depth_threshold:
        # Depth <= sqrt(k): use tree_cover_dfs - forming subtrees is efficient
        covering = tree_cover_dfs(Tree(tree, deep=True), num_robots)
    else:
        # Depth > threshold: traverse down until we find branching points
        # or subtrees with depth <= threshold
        covering = []

        def find_split_points(node, tree_ref, threshold):
            """Recursively find good split points in the tree."""
            subtree = tree_ref.subtree(node)
            subtree_depth = subtree.depth()

            # If this subtree is shallow enough, it becomes a covering chunk
            if subtree_depth <= threshold:
                return [set(subtree.nodes.keys())]

            children = tree_ref.children(node)
            if not children:
                # Leaf node
                return [{node}]

            if len(children) == 1:
                # Single child - continue traversing down (no split here)
                return find_split_points(children[0].identifier, tree_ref, threshold)
            else:
                # Multiple children - this is a branching point, split here
                result = []
                for child in children:
                    result.extend(find_split_points(child.identifier, tree_ref, threshold))
                return result

        covering = find_split_points(tree.root, tree, depth_threshold)

    # Go in dfs visit and explore each subtree in the cover recursively
    for cov_subtree in covering:
        # Update subtree as some vertices might have been already removed
        subtree = {v for v in cov_subtree if v in tree.nodes}
        if len(subtree) == 0:
            # All vertices already explored, skip this subtree
            continue

        subtree_root = find_root({v: 1 for v in subtree}, tree)
        if subtree_root is None or subtree_root not in tree.nodes:
            # Subtree root no longer in tree, skip
            continue

        # Go to root
        go_to_root = to_subtree_root(tree, num_robots, subtree_root, current_config)
        traversal.extend(list(go_to_root))

        # Get the actual subtree to explore
        subtree_tree = get_subtree(tree, subtree)
        if subtree_tree.size() == 0:
            continue

        # Recursively traverse the subtree
        dfs_recursive_exploration, to_delete = dfs_explore_internal(subtree_tree, num_robots,
                                                                     depth_constant=depth_constant, step_algorithm=step_algorithm,
                                                                     max_depth=max_depth)
        traversal.extend(list(dfs_recursive_exploration))

        # Squeeze at root - only if we have a valid last config
        if len(traversal) > 0:
            # Filter config to only include nodes still in tree
            last_config = {k: v for k, v in traversal[-1].items() if k in tree.nodes}
            if len(last_config) > 0:
                squeeze = squeeze_at_root(tree, last_config)
                traversal.extend(list(squeeze))

        # Delete vertices that were traversed
        for v in to_delete:
            if v in tree.nodes:
                tree.remove_node(v)

        # Gather back at subtree_root if it still exists
        if subtree_root in tree.nodes:
            # After deletion, current_config may reference deleted nodes
            # Find a valid starting point for gathering
            if len(traversal) > 0:
                last_cfg = traversal[-1]
                valid_nodes = [k for k in last_cfg.keys() if k in tree.nodes]
                if valid_nodes:
                    # Use the first valid node as starting point
                    start_node = valid_nodes[0]
                    current_config = {start_node: num_robots}
                else:
                    # All nodes deleted, use subtree_root
                    current_config = {subtree_root: num_robots}
            else:
                current_config = {subtree_root: num_robots}

            gathering = to_subtree_root(tree, num_robots, subtree_root, current_config)
            traversal.extend(list(gathering))
            current_config = {subtree_root: num_robots}
        else:
            # subtree_root was deleted, update current_config to a valid node
            if len(tree.nodes) > 0:
                current_config = {tree.root: num_robots}
            else:
                current_config = {}
    deleted = vertices - set(tree.nodes.keys())
    return tuple(traversal), deleted


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
    """Create a subtree containing exactly the nodes in subset, preserving tree structure."""
    if len(subset) == 0:
        return Tree()

    subtree_root = find_root({v: 1 for v in subset}, tree)
    subtree = Tree()
    subtree.create_node(identifier=subtree_root, tag=tree[subtree_root].tag)

    # Add nodes from subset in BFS order (to ensure parents are added before children)
    for v in tree.expand_tree(nid=subtree_root, mode=Tree.WIDTH):
        if v == subtree_root:
            continue
        if v in subset:
            parent = tree.parent(v).identifier
            # Only add if parent exists in subtree (maintains valid tree structure)
            if parent in subtree.nodes:
                subtree.create_node(identifier=v, tag=tree[v].tag, parent=parent)

    return subtree
